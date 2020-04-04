import numpy as np
from .detect import DSFDDetector, get_face_detections

detector = None


def detect_faces(
        image: np.ndarray,
        confidence_threshold: float,
        nms_iou_threshold=0.3,
        multiscale_detect=False,
        image_pyramid_detect=False,
        flip_detect=False):
    """
    Args:
        image: np.ndarray of shape [H, W, 3]
    Returns:
        boxes: np.ndarray of shape[N, 5] for N bounding boxes
            with [xmin, ymin, xmax, ymax, confidence]
    """
    global detector
    if detector is None:
        detector = DSFDDetector()
    return get_face_detections(
        detector,
        image,
        confidence_threshold,
        nms_iou_threshold,
        multiscale_detect,
        image_pyramid_detect,
        flip_detect
    )
