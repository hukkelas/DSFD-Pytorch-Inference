from .registry import build_from_cfg, Registry
from .base import Detector

available_detectors = [
    "DSFDDetector"
]
DETECTOR_REGISTRY = Registry("DETECTORS")


def build_detector(
        name: str = "DSFDDetector",
        confidence_threshold: float = 0.5,
        nms_iou_threshold: float = 0.3) -> Detector:
    assert name in available_detectors,\
        f"Detector not available. Chooce one of the following"+\
        ",".join(available_detectors)
    args = dict(
        type=name,
        confidence_threshold=confidence_threshold,
        nms_iou_threshold=nms_iou_threshold
    )
    detector = build_from_cfg(args, DETECTOR_REGISTRY)
    return detector
