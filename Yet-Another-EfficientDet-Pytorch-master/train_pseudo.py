# Author: Zylo117
# Editor: H2K

import argparse
import os
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2
import numpy as np
import pandas as pd
import re

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import invert_affine, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box, aspectaware_resize_padding
import ensemble_boxes
from tta_transform import *
from torchvision.ops.boxes import batched_nms
from itertools import product


import datetime
import traceback
import random

import yaml
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.autonotebook import tqdm

from backbone import EfficientDetBackbone
from efficientdet.dataset_aug import Resizer, Normalizer, Augmenter, collater, CSVDataset
from efficientdet.loss import FocalLoss
from utils.sync_batchnorm import patch_replication_callback
from utils.utils import replace_w_sync_bn, CustomDataParallel, get_last_weights, init_weights, boolean_string
import albumentations as A
# from utils.utils import display_gt
from train_aug import Params, ModelWithLoss, mixup, get_train_transforms

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
obj_list = ['wheat']
color_list = standard_to_bgr(STANDARD_COLORS)


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    # test
    parser.add_argument('-c', '--compound_coef', type=int, default=6, help='coefficients of efficientdet')
    
    parser.add_argument('--data_path', type=str, default='../datasets/global-wheat-detection', help='the root folder of dataset')
    parser.add_argument('-w', '--load_weights', nargs='+', type=str,
                        default='logs/global_wheat_detection_d6_mixup/efficientdet-d6_9_4000.pth',
                        help='trained weights')
    parser.add_argument('--iou_threshold', type=float, default=0.2)
    parser.add_argument('--threshold', type=float, default=0.43)
    parser.add_argument('--wbf_iou_threshold', type=float, default=0.55)
    parser.add_argument('--wbf_threshold', type=float, default=0.45)
    parser.add_argument('--force_size', type=int, default=512)

    # pseudo label
    parser.add_argument('--quantity', type=int, default=1)
    parser.add_argument('--confidence', type=float, default=0.6)

    # pseudo train
    parser.add_argument('-p', '--project', type=str, default='global-wheat-detection', help='project file that contains parameters')
    parser.add_argument('-n', '--num_workers', type=int, default=0, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=4, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=boolean_string, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=500, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--log_path', type=str, default='logs/pseudo_train/')
    parser.add_argument('-pw', '--load_pretrained_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/pseudo_train/')
    parser.add_argument('--debug', type=boolean_string, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')



    args = parser.parse_args()
    return args


def test(args):
    # set configs
    compound_coef = args.compound_coef
    force_input_size = 512  # set None to use default size
    img_path = os.path.join(args.data_path, 'test')
    image_ids = os.listdir(img_path)

    threshold = args.threshold
    iou_threshold = args.iou_threshold

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    # TTA setting
    tta_transforms = []
    for tta_combination in product([TTAHorizontalFlip(input_size), None],
                                   [TTAVerticalFlip(input_size), None],
                                   [TTARotate90(input_size), TTARotate180(input_size), TTARotate270(input_size), None],
                                   ):
        tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))


    #load model
    models = []
    for i in range(len(args.load_weights)):
        model = load_model(args, i)
        if use_cuda:
            model = model.cuda()
        if use_float16:
            model = model.half()
        models.append(model)

    # start prediction
    results = []
    with torch.no_grad():
        for image_id in image_ids:
            # read the image and resize to proper input size
            ori_imgs, framed_imgs, framed_metas = preprocess(os.path.join(img_path, image_id), max_size=input_size)

            if use_cuda:
                framed_imgs = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                framed_imgs = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

            framed_imgs= framed_imgs.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

            # tta predictions
            predictions = []
            for model in models:
                for transform in tta_transforms:
                    x = transform.batch_augment(framed_imgs.clone())

                    features, regression, classification, anchors = model(x)

                    regressBoxes = BBoxTransform()
                    clipBoxes = ClipBoxes()

                    # out: deaugmented bboxes list with shape(# of imgs, dict={rois,class_ids,scores})
                    out = postprocess(x, anchors, regression, classification,
                                    regressBoxes, clipBoxes,
                                    threshold, iou_threshold, transform)

                    predictions.append(out)

            # run wbf to fuse every image.
            wbf_out = []
            for i, _ in enumerate(framed_imgs):
                boxes, scores, labels = run_wbf(predictions, i, input_size, args.wbf_iou_threshold, args.wbf_threshold)
                boxes = boxes.round().astype(np.int32).clip(min=0, max=input_size-1)
                if len(boxes) != 0:
                    wbf_out.append({
                        'rois': boxes,
                        'class_ids': labels,
                        'scores': scores,
                    })
                else:
                    wbf_out.append({
                        'rois': np.array(()),
                        'class_ids': np.array(()),
                        'scores': np.array(()),
                    })

            # recover the predicted results to proper size corresponding to original input size
            wbf_out = invert_affine(framed_metas, wbf_out)
            results += wbf_out
            
    
    return results, image_ids


def run_wbf(predictions, image_index, image_size=512, iou_thr=0.55, skip_box_thr=0.45, weights=None):
    boxes = [(prediction[image_index]['rois']/(image_size-1)).tolist() for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]
    labels = [np.zeros(prediction[image_index]['scores'].shape[0]).astype(int).tolist() for prediction in predictions]
    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels.astype(int)


def load_model(args, idx=0):
    model = EfficientDetBackbone(compound_coef=args.compound_coef, num_classes=len(obj_list),
                                 ratios=anchor_ratios, scales=anchor_scales)
    model.load_state_dict(torch.load(args.load_weights[idx], map_location='cpu'))
    model.requires_grad_(False)
    model.eval()
    return model


def preprocess(*image_path, max_size=512):
    ori_imgs = [cv2.imread(img_path) for img_path in image_path]
    ori_imgs = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in ori_imgs]
    normalized_imgs = [(img.astype(np.float32) / 255) for img in ori_imgs]

    imgs_meta = [aspectaware_resize_padding(img, max_size, max_size,
                                            means=None) for img in normalized_imgs]
    framed_imgs = [img_meta[0] for img_meta in imgs_meta]
    framed_metas = [img_meta[1:] for img_meta in imgs_meta]

    return ori_imgs, framed_imgs, framed_metas


def postprocess(x, anchors, regression, classification, regressBoxes, clipBoxes, threshold, iou_threshold, transform):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, x)
    scores = torch.max(classification, dim=2, keepdim=True)[0]
    scores_over_thresh = (scores > threshold)[:, :, 0]
    out = []
    for i in range(x.shape[0]):
        if scores_over_thresh[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue

        classification_per = classification[i, scores_over_thresh[i, :], ...].permute(1, 0)
        transformed_anchors_per = transformed_anchors[i, scores_over_thresh[i, :], ...]
        scores_per = scores[i, scores_over_thresh[i, :], ...]
        scores_, classes_ = classification_per.max(dim=0)
        anchors_nms_idx = batched_nms(transformed_anchors_per, scores_per[:, 0], classes_, iou_threshold=iou_threshold)

        if anchors_nms_idx.shape[0] != 0:
            classes_ = classes_[anchors_nms_idx]
            scores_ = scores_[anchors_nms_idx]
            boxes_ = transformed_anchors_per[anchors_nms_idx, :]

            out.append({
                'rois': transform.deaugment_boxes(boxes_.cpu().numpy()),
                'class_ids': classes_.cpu().numpy(),
                'scores': scores_.cpu().numpy(),
            })
        else:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })

    return out


def display(preds, imgs, image_ids, imshow=True, imwrite=False, out_path='./'):
    if not os.path.exists(out_path):
        os.mkdir(out_path)

    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])
            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj,score=score,color=color_list[get_index_label(obj, obj_list)])

        if imshow:
            cv2.imshow(image_ids[i], imgs[i])
            cv2.waitKey(0)

        if imwrite:
            cv2.imwrite(os.path.join(out_path, image_ids[i]), imgs[i])


def writeSubmission(image_ids, preds):
    results = []

    for i in range(len(preds)):
        pred = preds[i]
        image_id = image_ids[i]
        scores = []
        boxes = []

        for j in range(len(pred['rois'])):
            x1, y1, x2, y2 = pred['rois'][j].astype(np.int)
            score = float(pred['scores'][j])
            scores.append(score)

            # each row: [image_id, predictionString] (predictionString = score x1 y1 w h score x1 y1 w h ...)
            box = x1, y1, x2 - x1, y2 - y1
            boxes.append(box)

        result = {
            'image_id': image_id.split('.')[0],
            'PredictionString': format_prediction_string(boxes, scores)
        }
        results.append(result)

    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    test_df.to_csv('submission.csv', index=False)
    test_df.head()


def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    return " ".join(pred_strings)


def pseudo_labeling(args, results, image_ids):
    testdf_pseudo = []

    for i in range(len(results)):
        if len(results[i]['rois']) == 0:
            continue

        pseudo = []
        scores = 0
        for j in range(len(results[i]['rois'])):
            x1, y1, x2, y2 = results[i]['rois'][j].astype(np.float)
            obj = obj_list[results[i]['class_ids'][j]]
            score = float(results[i]['scores'][j])
            scores += score
            result = {
                'image_id': 'nvnn'+image_ids[i].split('.')[0],
                'width': 1024,
                'height': 1024,
                'bbox': f'[{x1}, {y1}, {x2 - x1}, {y2 - y1}]', # x, y, w, h
                'source': 'nvnn'
            }
            pseudo.append(result)

        confidence = scores / len(pseudo)
        if confidence > args.confidence:
            for q in range(args.quantity):
                testdf_pseudo += pseudo


    test_df_pseudo = pd.DataFrame(testdf_pseudo, columns=['image_id', 'width', 'height', 'bbox', 'source'])

    train_df = pd.read_csv(os.path.join(args.data_path, 'train.csv'))
    print(train_df)
    print(test_df_pseudo)
    # train_df['x'] = -1
    # train_df['y'] = -1
    # train_df['w'] = -1
    # train_df['h'] = -1

    # def expand_bbox(x):
    #     r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    #     if len(r) == 0:
    #         r = [-1, -1, -1, -1]
    #     return r

    # train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
    # train_df.drop(columns=['bbox'], inplace=True)
    # train_df['x'] = train_df['x'].astype(np.float)
    # train_df['y'] = train_df['y'].astype(np.float)
    # train_df['w'] = train_df['w'].astype(np.float)
    # train_df['h'] = train_df['h'].astype(np.float)

    image_ids = train_df['image_id'].unique()
    valid_ids = image_ids[-300:]
    train_ids = image_ids[:-300]

    valid_df = train_df[train_df['image_id'].isin(valid_ids)]
    frames = [train_df, test_df_pseudo]
    train_df = pd.concat(frames, ignore_index=True)

    return train_df


def pseudo_train(opt, train_df):
    params = Params(f'projects/{opt.project}.yml')

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)

    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 512, 512, 1536, 1536]
    # training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
    #                            transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
    #                                                          Augmenter(),
    #                                                          Resizer(input_sizes[opt.compound_coef])]))
    training_set = CSVDataset(root_dir=opt.data_path, set=params.train_set,
                               transform=get_train_transforms(size=input_sizes[opt.compound_coef]), pseudo_csv=train_df)
    training_generator = DataLoader(training_set, **training_params)

    val_set = CSVDataset(root_dir=opt.data_path, set=params.val_set,
                          transform=get_train_transforms(size=input_sizes[opt.compound_coef]), pseudo_csv=train_df)
    val_generator = DataLoader(val_set, **val_params)

    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    # load last weights
    if opt.load_pretrained_weights is not None:
        if opt.load_pretrained_weights.endswith('.pth'):
            weights_path = opt.load_pretrained_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False

    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        print('> Use {0} gpus'.format(params.num_gpus))
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()

    svaeCKPT = 0

    num_iter_per_epoch = len(training_generator)

    try:
        print("> Start training ...")
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']

                    imgs, annot = mixup(imgs, annot) # 0.5 probility

                    # imgs_diplay = imgs.permute(0, 2, 3, 1).cpu().numpy()
                    # imgs_diplay = ((imgs_diplay * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255).astype(np.uint8)
                    # imgs_diplay = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs_diplay]
                    # display_gt(annot, imgs_diplay, imshow=False, imwrite=True)


                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()


                    optimizer.zero_grad()
                    cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % opt.save_interval == 0 and step > 0:
                        print(f'Saving ckpt efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth as last_ckpt.pth')
                        # save_checkpoint(model, 'last_ckpt.pth', opt)
                        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth', opt)
                        # print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))

            if epoch % opt.val_interval == 0:
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, opt.num_epochs, cls_loss, reg_loss, loss))
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)

                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    print(f'Saving ckpt efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth as last_ckpt.pth')
                    # save_checkpoint(model, 'last_ckpt.pth', opt)
                    save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth', opt)
                model.train()

                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                    break
    except KeyboardInterrupt:
        print(f'Saving ckpt efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth as last_ckpt.pth')
        # save_checkpoint(model, 'last_ckpt.pth', opt)
        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth', opt)
        writer.close()
    writer.close()


def save_checkpoint(model, name, opt):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(opt.saved_path, name))


if __name__ == '__main__':
    args = get_args()
    print("> Predict test images ...")
    results, image_ids = test(args)
    print("> Pseudo labeling ...")
    train_df = pseudo_labeling(args, results, image_ids)
    print("> Train test images ...")
    pseudo_train(args, train_df)