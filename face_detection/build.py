from typing import Optional

from face_detection.registry import build_from_cfg, Registry
from face_detection.base import Detector


available_detectors = [
    "DSFDDetector",
    "RetinaNetResNet50",
    "RetinaNetMobileNetV1"
]
DETECTOR_REGISTRY = Registry("DETECTORS")


def build_detector(
        name: str = "DSFDDetector",
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.3,
        device: str = "cpu",
        max_resolution: int = None,
        fp16_inference: bool = False,
        clip_boxes: bool = False,
        model_weights: Optional[str] = None,
        ) -> Detector:
    assert name in available_detectors,\
        f"""Detector not available. 
        Choose one of the following {','.join(available_detectors)}
        """
        
    args = dict(
        type=name,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold,
        device=device,
        max_resolution=max_resolution,
        fp16_inference=fp16_inference,
        clip_boxes=clip_boxes,
        model_weights=model_weights,
    )
    detector = build_from_cfg(args, DETECTOR_REGISTRY)
    return detector
