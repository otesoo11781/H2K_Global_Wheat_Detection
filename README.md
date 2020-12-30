# H2K_Global_Wheat_Detection
This is a implementation for final project of Visual Recognition using Deep Learning - Global Wheat Detection, which is a Kaggle competition.

The goal of the competition is to detect wheat heads in given images.

We retrain EfficientDet-D6 pretrained on MSCOCO with multiple data augmentations.

In addition, we apply several model tricks, including weighted boxes fusion (WBF) over testing time augmentation (TTA) and pseudo labeling. 

The details please refer to kaggle website [Global Wheat Detection](https://www.kaggle.com/c/global-wheat-detection/overview).

**Important: the code is based on [Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch), which is an open source implementation of EfficientDet in Pytorch.**

## Hardware
The following specs were used to train and test the model:
- Ubuntu 16.04 LTS
- 2x RTX TITAN with CUDA=10.1

## Reproducing Submission
To reproduct my submission, do the following steps:
1. [Installation](#installation)
2. [Dataset Preparation](#dataset-preparation)
3. [Transfer Training](#transfer-training)
4. [Inference](#inference)
5. [Make Submission](#make-submission)

## Installation
Run the following command to install our implementation:

```shell
# install the requirements
pip install pycocotools numpy opencv-python tqdm tensorboard tensorboardX pyyaml webcolors
pip install torch==1.4.0
pip install torchvision==0.5.0
pip install ensemble-boxes
pip install -U albumentations
```

## Dataset Preparation
Download the dataset from the [Global Wheat Detection](https://www.kaggle.com/c/global-wheat-detection/data) on Kaggle website.

Then, unzip them and put them under the **./datasets/** directory.

Hence, the data directory is structured as:
```
./datasets/
  +- test/
  |  +- 2fd875eaa.jpg
  |  +- ....jpg 
  +- train/
  |  +- 00b5c6764.jpg
  |  +- ....jpg
  +- sample_submission.csv
  +- train.csv
```

## Transfer Training
**Important: This step is optional. If you don't want to retrain the MSCOCO pretrained model, please download [our trained weights](https://drive.google.com/file/d/1pl82ZLSlgvjJe03MZvMJZ6Iqoz1KdsVr/view?usp=sharing).**

- **efficientdet-d6_204_57500.pth**: our weights trained on global wheat detection dataset with our training strategy. 

Then, move **efficientdet-d6_204_57500.pth** to the **./Yet-Another-EfficientDet-Pytorch-master/logs/aug_mix_cut/global-wheat-detection/** directory.

Hence, the weights directory is structured as:
```
./Yet-Another-EfficientDet-Pytorch-master/
  |  +- logs/
     |  +- aug_mix_cut/
        |  +- global-wheat-detection/
           |  +- efficientdet-d6_204_57500.pth
```

### Retrain the MS COCO pretrained model on the given dataset (optional)
P.S. If you don't want to spend a half day training a model, you can skip this step and just use the **efficientdet-d6_204_57500.pth** I provided to inference. 

Now, let's transferly train the EfficientDet-D6 on global wheat dateset.

0. please download [MS COCO pretrained weights](https://drive.google.com/file/d/1TBx_8mX_zmDghYv6cIFiJC7ntCB9qbFp/view?usp=sharing)

1. move **efficientdet-d6.pth** to the **./Yet-Another-EfficientDet-Pytorch-master/pretrained_weights/** directory. 

2. please check your current directory is ./Yet-Another-EfficientDet-Pytorch-master.

3. run the following training command:

```
$ python train_aug.py -c 6 --batch_size 12 --num_epochs 85 --le 1e-4 -w pretrained_weights/efficientdet-d6.pth
```

In our report, we first train our model without cutmix for 80 epochs, and then retrain with cutmix for 5 epochs.

Finally, we can find the final weights in **./Yet-Another-EfficientDet-Pytorch-master/logs/aug_mix_cut/global-wheat-detection/** directory.


## Inference
With the testing dataset and trained model, you can run the following commands to obtain the predicted results:

1. please check your current directory is ./Yet-Another-EfficientDet-Pytorch-master.

2. run the pseudo labeling training script:

```
$ python train_pseudo.py --data_path ../datasets/global-wheat-detection -c 6 -w ./logs/aug_mix_cut/global-wheat-detection/efficientdet-d6_204_57500_mix_cut.pth --threshold 0.43 -p global_wheat_detection -n 0 --batch_size 4 --num_epochs 2 -pw ./logs/aug_mix_cut/global-wheat-detection/efficientdet-d6_204_57500_mix_cut.pth --wbf_threshold 0.45 --wbf_iou_threshold 0.55 --quantity 1 --confidence 0.3
```

3. do the final prediction by following command:

```
$ python wheat_test.py --data_path ../datasets/global-wheat-detection -c 6 -w ./logs/pseudo_train/global-wheat-detection/last_ckpt.pth --threshold 0.43 --wbf_threshold 0.45 --wbf_iou_threshold 0.55
```

After that, you will get final detection result (**./Yet-Another-EfficientDet-Pytorch-master/submission.csv**).


## Make Submission
1. submit pseudo.ipynb to kaggle competition.

The final scores are 0.7261 0.6433 on public and private leaderboard, respectively.


