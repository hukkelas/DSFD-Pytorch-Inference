# Dual Shot Face Detection - Pytorch Inference Code 
A High-Performance Pytorch Implementation of the paper "[DSFD: Dual Shot Face Detector" (CVPR 2019).](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_DSFD_Dual_Shot_Face_Detector_CVPR_2019_paper.pdf).

![](example_det.jpg)

**NOTE** This implementation is based on the [original source code](https://github.com/TencentYoutuResearch/FaceDetection-DSFD)  released by the authors of the paper. This repo can only be used for inference of their model based on ResNet-152 and all training scripts are removed. If you want to finetune the model, we recommend you to use the original source code.

## Requirements

- Python >= 3.6 
- Pytorch >= 1.1
- Torchvision >= 0.3.0
- OpenCV

If you are familiar with Docker, we recommend you to use our custom [Dockerfile](Dockerfile) to set up your environment.

## Improvements from original source code

- Removal of all unnecessary files for training / loading VGG models. 
- Improve the inference time by about 30x (from ~6s to 0.2) with rough estimates using `time` (Measured on a V100-32GB GPU).

The main improvements in inference time comes from:

- Replacing non-maximum-suppression to a [highly optimized torchvision version](https://github.com/pytorch/vision/blob/19315e313511fead3597e23075552255d07fcb2a/torchvision/ops/boxes.py#L5)
- Refactoring `init_priors`in the [SSD model](dsfd/face_ssd.py) to cache previous prior sizes (no need to generate this per forward pass).
- Refactoring the forward pass in `Detect` in [`utils.py`](dsfd/utils.py) to perform confidence thresholding before non-maximum suppression
- Minor changes in the forward pass to use pytorch 1.0 features 

## Getting started

1. Download the original weights file from the [original source code repo](https://github.com/TencentYoutuResearch/FaceDetection-DSFD). Place this into the path `dsfd/weights/`
2. Run
```
python3 test.py
```
This will look for images in the `images/` folder, and save the results in the same folder with an ending `_out.jpg`

## Simple API
To perform detection you can simple use the following lines:

```python
import cv2
from dsfd.detect import DSFDDetector
weight_path = "dsfd/weights/WIDERFace_DSFD_RES152.pth"
im = cv2.imread("path_to_im.jpg")
detector = DSFDDetector(weight_path)
detections = detector.detect_face(im, confidence_threshold=.5, shrink=1.0)
```

This will return a tensor with shape `[N, 5]`, where N is number of faces and the five elements are `[xmin, ymin, xmax, ymax, detection_confidence]`

## Inference time

This is **very roughly** estimated on a 1024x687 image. The reported time is the average over 100 runs. (With no cudnn benchmarking and no fp16 computation).


| **Device**                                       | **MS**     |
|----------------------------------------------|--------|
| CPU (MacOS Mid '14 15-Inch, Intel 2.2GHz i7) | 17,496 |
| GPU (1x NVIDIA V100-32GB)                    | 100    |
|                                              |        |

## Replicate WIDER-Face performance
For their results on the WIDER-Face dataset, they used detection over several image scales. This is replicated in the function `get_face_detections`. 

`use_multiscale_detect` and `use_image_pyramid_detect` are very slow. In most cases, having all to `False` works well and is the fastest method. However, if you want to reproduce any of their results in the paper, we recommend you to turn all to `True`

**NOTE**: We have (yet) not tried to replicate their results on any of the datasets presented in the paper.

## Citation
If you find this code useful, remember to cite the original authors:
```
@inproceedings{li2018dsfd,
  title={DSFD: Dual Shot Face Detector},
  author={Li, Jian and Wang, Yabiao and Wang, Changan and Tai, Ying and Qian, Jianjun and Yang, Jian and Wang, Chengjie and Li, Jilin and Huang, Feiyue},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
