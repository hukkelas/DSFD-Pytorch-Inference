# State of the Art Face Detection in Pytorch with DSFD and RetinaFace

This repository includes:
- A High-Performance Pytorch Implementation of the paper "[DSFD: Dual Shot Face Detector" (CVPR 2019).](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_DSFD_Dual_Shot_Face_Detector_CVPR_2019_paper.pdf) adapted from the [original source code](https://github.com/TencentYoutuResearch/FaceDetection-DSFD).
- Lightweight single-shot face detection from the paper [RetinaFace: Single-stage Dense Face Localisation in the Wild](https://arxiv.org/abs/1905.00641) adapted from https://github.com/biubug6/Pytorch_Retinaface.

![](example_det.jpg)

**NOTE** This implementation can only be used for inference of a selection of models and all training scripts are removed. If you want to finetune any models, we recommend you to use the original source code.

## Install

You can install this repository with pip (requires python>=3.6);

```bash
pip install git+https://github.com/hukkelas/DSFD-Pytorch-Inference.git
```

You can also install with the `setup.py`

```bash
python3 setup.py install
```

## Model weights

In this table you can see the urls for the different models implemented. We recommend to download them to local before do an inference:

|Model|Weights url|
|:-:|:-:|
|RetinaNetMobileNetV1|[Link](https://raw.githubusercontent.com/hukkelas/DSFD-Pytorch-Inference/master/RetinaFace_mobilenet025.pth)|
|RetinaNetResNet50|[Link](https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/8dd81669-eb84-4520-8173-dbe49d72f44cb2eef6da-3983-4a12-9085-d11555b93842c19bdf27-b924-4214-9381-e6cac30b87cf)|
|DSFDDetector|[Link 1](https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/61be4ec7-8c11-4a4a-a9f4-827144e4ab4f0c2764c1-80a0-4083-bbfa-68419f889b80e4692358-979b-458e-97da-c1a1660b3314) - [Link 2](https://drive.google.com/uc?id=1WeXlNYsM6dMP3xQQELI-4gxhwKUQxc3-&export=download)|



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

This is **very roughly** estimated on a 1024x687 image. The reported time is the average over 1000 forward passes on a single image. (With no cudnn benchmarking and no fp16 computation).


| | DSFDDetector | RetinaNetResNet50 | RetinaNetMobileNetV1 |
| -|-|-|-|
| CPU (Intel 2.2GHz i7) *| 17,496 ms (0.06 FPS) | 2970ms (0.33 FPS) | 270ms (3.7 FPS) | 
| NVIDIA V100-32GB | 100ms (10 FPS) | | |
| NVIDIA GTX 1060 6GB | 341ms (2.9 FPS) | 76.6ms (13 FPS) | 48.2ms (20.7 FPS) | 
| NVIDIA T4 16 GB | 482 ms (2.1 FPS) | 181ms (5.5 FPS) | 178ms (5.6 FPS) |

*Done over 100 forward passes on a MacOS Mid 2014, 15-Inch.

## Changelog 
  - September 1st 2020: added support for fp16/mixed precision inference
  - September 24th 2020: added support for TensorRT.


## TensorRT Inference (Experimental)
You can run RetinaFace ResNet-50 with TensorRT:

```python
from face_detection.retinaface.tensorrt_wrap import TensorRTRetinaFace

inference_imshape =(480, 640) # Input to the CNN
input_imshape = (1080, 1920) # Input for original video source
detector = TensorRTRetinaFace(input_imshape, imshape)
boxes, landmarks, scores = detector.infer(image)

```

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
