# Adapted from https://github.com/biubug6/Pytorch_Retinaface
# Original license: MIT
import torch
import numpy as np
from .. import torch_utils
import typing
from .models.retinaface import RetinaFace
from ..box_utils import batched_decode
from .utils import decode_landm
from .config import cfg_mnet, cfg_re50
from .prior_box import PriorBox
from torch.hub import load_state_dict_from_url
from torchvision.ops import nms
from ..base import Detector
from ..build import DETECTOR_REGISTRY


class RetinaNetDetector(Detector):

    def __init__(
            self,
            model: str,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)
        if model == "mobilenet":
            cfg = cfg_mnet
            state_dict = load_state_dict_from_url(
                "https://folk.ntnu.no/haakohu/RetinaFace_mobilenet025.pth",
                map_location=torch_utils.get_device()
            )
        else:
            assert model == "resnet50"
            cfg = cfg_re50
            state_dict = load_state_dict_from_url(
                "https://folk.ntnu.no/haakohu/RetinaFace_ResNet50.pth",
                map_location=torch_utils.get_device()
            )
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        net = RetinaFace(cfg=cfg)
        net.eval()
        net.load_state_dict(state_dict)
        self.cfg = cfg
        self.net = net.to(self.device)
        self.mean = np.array([104, 117, 123], dtype=np.float32)
        self.prior_box_cache = {}

    def batched_detect_with_landmarks(
            self, image: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Takes N images and performs and returns a set of bounding boxes as
            detections
        Args:
            image (np.ndarray): shape [N, height, width, 3]
        Returns:
            np.ndarray: shape [N, 5] with (xmin, ymin, xmax, ymax, score)
            np.ndarray: shape [N, 5, 2] with 5 landmarks with (x, y)
        """
        image = image.astype(np.float32) - self.mean
        image = np.moveaxis(image, -1, 1)
        image = torch.from_numpy(image)
        orig_shape = image.shape[2:]
        image = self.resize(image, 1).to(self.device)
        boxes, landms = self._detect(image, return_landmarks=True)
        scores = boxes[:, :, -1]
        boxes = boxes[:, :, :-1]
        final_output_box = []
        final_output_landmarks = []
        for i in range(len(boxes)):
            boxes_ = boxes[i]
            landms_ = landms[i]
            scores_ = scores[i]
            # Confidence thresholding
            keep_idx = scores_ >= self.confidence_threshold
            boxes_ = boxes_[keep_idx]
            scores_ = scores_[keep_idx]
            landms_ = landms_[keep_idx]
            # Non maxima suppression
            keep_idx = nms(
                boxes_, scores_, self.nms_iou_threshold)
            boxes_ = boxes_[keep_idx]
            scores_ = scores_[keep_idx]
            landms_ = landms_[keep_idx]
            # Scale boxes
            height, width = orig_shape
            if self.clip_boxes:
                boxes_ = boxes_.clamp(0, 1)
            boxes_[:, [0, 2]] *= width
            boxes_[:, [1, 3]] *= height

            # Scale landmarks
            landms_ = landms_.cpu().numpy().reshape(-1, 5, 2)
            landms_[:, :, 0] *= width
            landms_[:, :, 1] *= height
            dets = torch.cat(
                (boxes_, scores_.view(-1, 1)), dim=1).cpu().numpy()
            final_output_box.append(dets)
            final_output_landmarks.append(landms_)
        return final_output_box, final_output_landmarks

    @torch.no_grad()
    def _detect(
            self, image: np.ndarray,
            return_landmarks=False) -> np.ndarray:
        """Batched detect
        Args:
            image (np.ndarray): shape [N, H, W, 3]
        Returns:
            boxes: list of length N with shape [num_boxes, 5] per element
        """
        image = image[:, [2, 1, 0]]
        with torch.cuda.amp.autocast(enabled=self.fp16_inference):
            loc, conf, landms = self.net(image)  # forward pass
            scores = conf[:, :, 1:]
            height, width = image.shape[2:]
            if image.shape[2:] in self.prior_box_cache:
                priors = self.prior_box_cache[image.shape[2:]]
            else:
                priorbox = PriorBox(
                    self.cfg, image_size=(height, width))
                priors = priorbox.forward()
                self.prior_box_cache[image.shape[2:]] = priors
            priors = torch_utils.to_cuda(priors, self.device)
            prior_data = priors.data
            boxes = batched_decode(loc, prior_data, self.cfg['variance'])
            boxes = torch.cat((boxes, scores), dim=-1)
        if return_landmarks:
            landms = decode_landm(landms, prior_data, self.cfg['variance'])
            return boxes, landms
        return boxes


@DETECTOR_REGISTRY.register_module
class RetinaNetResNet50(RetinaNetDetector):

    def __init__(self, *args, **kwargs):
        super().__init__("resnet50", *args, **kwargs)


@DETECTOR_REGISTRY.register_module
class RetinaNetMobileNetV1(RetinaNetDetector):

    def __init__(self, *args, **kwargs):
        super().__init__("mobilenet", *args, **kwargs)
