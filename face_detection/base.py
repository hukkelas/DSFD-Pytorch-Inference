import numpy as np
from abc import ABC


class Detector(ABC):

    def __init__(
            self,
            confidence_threshold: float,
            nms_iou_threshold: float):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def detect(
            self, image: np.ndarray) -> np.ndarray:
        """Takes an RGB image and performs and returns a set of bounding boxes as
            detections
        Args:
            image (np.ndarray): shape [height, width, 3]
        Returns:
            np.ndarray: shape [N, 5] with (xmin, ymin, xmax, ymax, score)
        """
        raise NotImplementedError
