import numpy as np
import torch
import typing
from .torch_utils import get_device
from abc import ABC, abstractmethod
from torchvision.ops import nms
from .box_utils import scale_boxes


class Detector(ABC):

    def __init__(
            self,
            confidence_threshold: float,
            nms_iou_threshold: float,
            device=get_device()):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device
        self.mean = np.array(
            [123, 117, 104], dtype=np.float32).reshape(1, 1, 1, 3)

    def detect(
            self, image: np.ndarray, shrink=1.0) -> np.ndarray:
        """Takes an RGB image and performs and returns a set of bounding boxes as
            detections
        Args:
            image (np.ndarray): shape [height, width, 3]
        Returns:
            np.ndarray: shape [N, 5] with (xmin, ymin, xmax, ymax, score)
        """
        image = image[None]
        boxes = self.batched_detect(image, shrink)
        return boxes[0]

    @abstractmethod
    def _detect(self, image: torch.Tensor) -> torch.Tensor:
        """Takes N RGB image and performs and returns a set of bounding boxes as
            detections
        Args:
            image (torch.Tensor): shape [N, height, width, 3]
        Returns:
            torch.Tensor: of shape [N, B, 5] with (xmin, ymin, xmax, ymax, score)
        """
        raise NotImplementedError

    def filter_boxes(self, boxes: torch.Tensor) -> typing.List[np.ndarray]:
        """Performs NMS and score thresholding

        Args:
            boxes (torch.Tensor): shape [N, B, 5] with (xmin, ymin, xmax, ymax, score)
        Returns:
            list: N np.ndarray of shape [B, 5]
        """
        final_output = []
        for i in range(len(boxes)):
            scores = boxes[i, :,  4]
            keep_idx = scores >= self.confidence_threshold
            boxes_ = boxes[i, keep_idx, :-1]
            scores = scores[keep_idx]
            if scores.dim() == 0:
                final_output.append(torch.empty(0, 5))
                continue
            keep_idx = nms(boxes_, scores, self.nms_iou_threshold)
            scores = scores[keep_idx].view(-1, 1)
            boxes_ = boxes_[keep_idx].view(-1, 4)
            output = torch.cat((boxes_, scores), dim=-1)
            final_output.append(output)
        return final_output

    @torch.no_grad()
    def batched_detect(
            self, image: np.ndarray, shrink=1.0) -> typing.List[np.ndarray]:
        """Takes N RGB image and performs and returns a set of bounding boxes as
            detections
        Args:
            image (np.ndarray): shape [height, width, 3]
        Returns:
            np.ndarray: a list with N set of bounding boxes of
                shape [B, 5] with (xmin, ymin, xmax, ymax, score)
        """
        assert image.dtype == np.uint8
        height, width = image.shape[1:3]
        image = image.astype(np.float32) - self.mean
        image = np.moveaxis(image, -1, 1)
        image = torch.from_numpy(image)
        image = torch.nn.functional.interpolate(image, scale_factor=shrink)
        image = image.to(self.device)
        boxes = self._detect(image)
        boxes = self.filter_boxes(boxes)
        boxes = [scale_boxes((height, width), box).cpu().numpy() for box in boxes]
        
        return boxes
