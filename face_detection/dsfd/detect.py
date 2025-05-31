import torch
import numpy as np
import typing

from face_detection import torch_utils
from face_detection.dsfd.face_ssd import SSD
from face_detection.dsfd.config import resnet152_model_config
from face_detection.base import Detector
from face_detection.build import DETECTOR_REGISTRY


@DETECTOR_REGISTRY.register_module
class DSFDDetector(Detector):

    def __init__(
            self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        state_dict = torch_utils.load_weights(self.model_weights)
        self.net = SSD(resnet152_model_config)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.net = self.net.to(self.device)

    @torch.no_grad()
    def _detect(self, x: torch.Tensor,) -> typing.List[np.ndarray]:
        """Batched detect
        Args:
            image (np.ndarray): shape [N, H, W, 3]
        Returns:
            boxes: list of length N with shape [num_boxes, 5] per element
        """
        # Expects BGR
        x = x[:, [2, 1, 0], :, :]
        with torch.cuda.amp.autocast(enabled=self.fp16_inference):
            boxes = self.net(
                x, self.confidence_threshold, self.nms_iou_threshold
            )
        return boxes
