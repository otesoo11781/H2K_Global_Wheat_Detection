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

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import invert_affine, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box, aspectaware_resize_padding
import ensemble_boxes
from tta_transform import *
from torchvision.ops.boxes import batched_nms
from itertools import product

# replace this part with your project's anchor config
anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
obj_list = ['wheat']
color_list = standard_to_bgr(STANDARD_COLORS)


def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-c', '--compound_coef', type=int, default=6, help='coefficients of efficientdet')
    
    parser.add_argument('--data_path', type=str, default='../datasets/global_wheat_detection', help='the root folder of dataset')
    parser.add_argument('-w', '--load_weights', nargs='+', type=str,
                        default='logs/global_wheat_detection_d6_mixup/efficientdet-d6_9_4000.pth',
                        help='trained weights')
    parser.add_argument('--saved_path', type=str, default='./image_results')
    parser.add_argument('--iou_threshold', type=float, default=0.2)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--wbf_iou_threshold', type=float, default=0.44)
    parser.add_argument('--wbf_threshold', type=float, default=0.43)
    parser.add_argument('--force_size', type=int, default=512)

    args = parser.parse_args()
    return args


def test(args):
    # set configs
    compound_coef = args.compound_coef
    force_input_size = args.force_size  # set None to use default size
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

            # ori_imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in ori_imgs]
            # display(wbf_out, ori_imgs, [image_id], imshow=False, imwrite=True, out_path=args.saved_path)

        writeSubmission(image_ids, results)




def run_wbf(predictions, image_index, image_size=512, iou_thr=0.44, skip_box_thr=0.43, weights=None):
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


if __name__ == '__main__':
    args = get_args()
    test(args)
