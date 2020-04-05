# State of the Art Face Detection in Pytorch with DSFD and RetinaFace

This repository includes:
- A High-Performance Pytorch Implementation of the paper "[DSFD: Dual Shot Face Detector" (CVPR 2019).](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_DSFD_Dual_Shot_Face_Detector_CVPR_2019_paper.pdf) adapted from the [original source code](https://github.com/TencentYoutuResearch/FaceDetection-DSFD).
- Lightweight single-shot face detection from the paper [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641) adapted from https://github.com/biubug6/Pytorch_Retinaface.

![](example_det.jpg)

**NOTE** This implementation can only be used for inference of a selection of models and all training scripts are removed. If you want to finetune any models, we recommend you to use the original source code.

## Install

You can install this repository with pip (requires python>=3.6);

```bash
pip install face_detection
```

You can also install with the `setup.py`

```bash
python3 setup.py install
```

## Getting started
Run
```
python3 test.py
```
This will look for images in the `images/` folder, and save the results in the same folder with an ending `_out.jpg`

## Simple API
To perform detection you can simple use the following lines:

```python
import cv2
import face_detection
print(face_detection.available_detectors)
detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
# BGR to RGB
im = cv2.imread("path_to_im.jpg")[:, :, ::-1]

detections = detector.detect(im)
```

This will return a tensor with shape `[N, 5]`, where N is number of faces and the five elements are `[xmin, ymin, xmax, ymax, detection_confidence]`

### Batched inference

```python
import numpy as np
import face_detection
print(face_detection.available_detectors)
detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)
# [batch size, height, width, 3]
images_dummy = np.zeros((2, 512, 512, 3))

detections = detector.batched_detect(im)
```


## Improvements

### Difference from DSFD
For the original source code, see [here](https://github.com/TencentYoutuResearch/FaceDetection-DSFD).
- Removal of all unnecessary files for training / loading VGG models. 
- Improve the inference time by about 30x (from ~6s to 0.2) with rough estimates using `time` (Measured on a V100-32GB GPU).

The main improvements in inference time comes from:

- Replacing non-maximum-suppression to a [highly optimized torchvision version](https://github.com/pytorch/vision/blob/19315e313511fead3597e23075552255d07fcb2a/torchvision/ops/boxes.py#L5)
- Refactoring `init_priors`in the [SSD model](dsfd/face_ssd.py) to cache previous prior sizes (no need to generate this per forward pass).
- Refactoring the forward pass in `Detect` in [`utils.py`](dsfd/utils.py) to perform confidence thresholding before non-maximum suppression
- Minor changes in the forward pass to use pytorch 1.0 features 

### Difference from RetinaFace
For the original source code, see [here](https://github.com/biubug6/Pytorch_Retinaface).

We've done the following improvements:
- Remove gradient computation for inference (`torch.no_grad`).
- Replacing non-maximum-suppression to a [highly optimized torchvision version](https://github.com/pytorch/vision/blob/19315e313511fead3597e23075552255d07fcb2a/torchvision/ops/boxes.py#L5)

## Inference time

This is **very roughly** estimated on a 1024x687 image. The reported time is the average over 100 runs. (With no cudnn benchmarking and no fp16 computation).


|Model| **Device** | **MS** |
|-|----------------------------------------------|--------|
|DSFDDetector| CPU (MacOS Mid '14 15-Inch, Intel 2.2GHz i7) | 17,496 |
|DSFDDetector| GPU (1x NVIDIA V100-32GB)|100|
|RetinaNetResNet50| CPU (MacOS Mid '14 15-Inch, Intel 2.2GHz i7) |3428|
|RetinaNetResNet50| GPU (1x NVIDIA V100-32GB)||
|RetinaNetMobileNetV1| CPU (MacOS Mid '14 15-Inch, Intel 2.2GHz i7)|281|
|RetinaNetMobileNetV1| GPU (1x NVIDIA V100-32GB)||




## Citation
If you find this code useful, remember to cite the original authors:
```
@inproceedings{li2018dsfd,
  title={DSFD: Dual Shot Face Detector},
  author={Li, Jian and Wang, Yabiao and Wang, Changan and Tai, Ying and Qian, Jianjun and Yang, Jian and Wang, Chengjie and Li, Jilin and Huang, Feiyue},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}

@inproceedings{deng2019retinaface,
  title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
  author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
  booktitle={arxiv},
  year={2019}

```
