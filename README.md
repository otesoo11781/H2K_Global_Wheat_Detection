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

Then, unzip them and put them under the **./dataset/** directory.

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
**Important: This step is optional. If you don't want to retrain the MSCOCO pretrained model, please download [my trained weights]().**

- **latest.pth**: my weights trained on tiny PASCAL VOC dataset (provided by TAs) with 24 epochs. 

Then, move **latest.pth** to the **./mmdetection/work_dirs/cascade_mask_rcnn_resnest/** directory.

Hence, the weights directory is structured as:
```
./mmdetection/
  +- work_dirs/
  |  +- cascade_mask_rcnn_resnest/
     |  +- latest.pth
```

### Retrain the ImageNet pretrained model on the given dataset (optional)
P.S. If you don't want to spend a half day training a model, you can skip this step and just use the **latest.pth** I provided to inference. 

Now, let's transferly train the Cascade Mask RCNN + ResNeSt on tiny PASCAL VOC dataset:

1. please ensure ./mmdetection/configs/myconfigs/cascade_mask_rcnn_resnest.py exists.

2. please check your current directory is ./mmdetection.

3. run the following training command (the last argument "2" means the number of gpus):

```
bash ./tools/dist_train.sh configs/myconfigs/cascade_mask_rcnn_resnest.py 2
```

It takes about 13 hours to train the model on 2 RTX 2080 GPUs.

Finally, we can find the final weights **latest.pth** in **./mmdetection/work_dirs/cascade_mask_rcnn_resnest/** directory.


## Inference
With the testing dataset and trained model, you can run the following commands to obtain the predicted results:

1. please check your current directory is ./mmdetection.

2. run the testing bash script (the third argument "2" means the number of gpus):

```
./tools/dist_test.sh configs/myconfigs/cascade_mask_rcnn_resnest.py ./work_dirs/cascade_mask_rcnn_resnest/latest.pth 2 --format-only --options "jsonfile_prefix=./0856610"
```

After that, you will get final segmentation result (**./mmdetection/0856610.segm.json**).


## Make Submission
1. rename the 0856610.segm.json as 0856610.json

2. submit **0856610.json** to [here](https://drive.google.com/drive/folders/1VhuHvCyz2CH4yzDreyVTwhZiOFbQB09B).

**Note**: The repo has provided **mAP_0.38246_0856610.json** which is my submission of predicted segmentation result with **0.38246 mAP**.


