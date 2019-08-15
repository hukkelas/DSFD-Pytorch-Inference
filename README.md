# Dual Shot Face Detection - Pytorch Inference Code 
A High-Performance Pytorch Implementation of the paper "[DSFD: Dual Shot Face Detector" (CVPR 2019).](http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_DSFD_Dual_Shot_Face_Detector_CVPR_2019_paper.pdf).
![](example_det.jpg)
**NOTE** This implementation is based on the [original source code](https://github.com/TencentYoutuResearch/FaceDetection-DSFD)  released by the authors of the paper. This repo can only be used for inference of their model based on ResNet-152 and all training scripts are removed. If you want to finetune the model, we recommend you to use the original source code.

## Requirements

- Python >= 3.6 
- Pytorch >= 1.1
- Torchvision >= 0.3.0
- OpenCV

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

## Inference hyperparameters
There are several parameters to set for inference. By default, the values are:
```
confidence_threshold = .3
nms_iou_threshold = .3
use_multiscale_detect = False
use_flip_detect = False
use_image_pyramid_detect = False
```
`use_multiscale_detect` and `use_image_pyramid_detect` are very slow. In most cases, having all to `False` works well and is the fastest method. However, if you want to reproduce any of their results in the paper, we recommend you to turn all to `True`


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
