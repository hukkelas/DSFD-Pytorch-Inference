# Adapted from https://github.com/biubug6/Pytorch_Retinaface
# Original license: MIT
import torch
import numpy as np
from .. import torch_utils
import typing
from .models.retinaface import RetinaFace
from ..box_utils import decode
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
            confidence_threshold: float,
            nms_iou_threshold: float):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
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
        net = torch_utils.to_cuda(net)

        self.cfg = cfg
        self.net = net
        self.mean = np.array([123, 117, 104], dtype=np.float32)

    def detect(
            self, image: np.ndarray) -> np.ndarray:
        # Expects BGR
        image = image[:, :, ::-1]
        detections, landmarks = self._detect(image)
        return detections

    def detect_with_landmarks(
            self, image: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        """Takes N images and performs and retunrs a set of bounding boxes as
            detections
        Args:
            image (np.ndarray): shape [N, height, width, 3]
        Returns:
            np.ndarray: shape [N, 5] with (xmin, ymin, xmax, ymax, score)
            np.ndarray: shape [N, 5, 2] with 5 landmarks with (x, y)
        """
        return self._detect(image)

    @torch.no_grad()
    def _detect(
            self, image: np.ndarray) -> np.ndarray:
        img = image
        height, width = img.shape[:2]
        assert img.dtype == np.uint8
        assert img.shape[-1] == 3
        img = img.astype(np.float32) - self.mean
        img = torch_utils.image_to_torch(
            img, cuda=True)
        loc, conf, landms = self.net(img)  # forward pass
        scores = conf.squeeze(0)[:, 1]
        loc = loc.squeeze(0)
        landms = landms.squeeze(0)
        priorbox = PriorBox(
            self.cfg, image_size=(height, width))
        priors = priorbox.forward()
        priors = torch_utils.to_cuda(priors)
        prior_data = priors.data
        boxes = decode(loc, prior_data, self.cfg['variance'])
        landms = decode_landm(landms, prior_data, self.cfg['variance'])
        # Confidence thresholding
        keep_idx = scores >= self.confidence_threshold
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        landms = landms[keep_idx]
        # Non maxima suppression
        keep_idx = nms(
            boxes, scores, self.nms_iou_threshold)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        landms = landms[keep_idx]
        # Scale boxes
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        # Scale landmarks
        landms = landms.cpu().numpy().reshape(-1, 5, 2)
        landms[:, :, 0] *= width
        landms[:, :, 1] *= height
        dets = torch.cat(
            (boxes, scores.view(-1, 1)), dim=1).cpu().numpy()

        return dets, landms


@DETECTOR_REGISTRY.register_module
class RetinaNetResNet50(RetinaNetDetector):

    def __init__(self, *args, **kwargs):
        super().__init__("resnet50", *args, **kwargs)


@DETECTOR_REGISTRY.register_module
class RetinaNetMobileNetV1(RetinaNetDetector):

    def __init__(self, *args, **kwargs):
        super().__init__("mobilenet", *args, **kwargs)
