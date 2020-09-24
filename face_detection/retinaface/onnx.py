# Adapted from https://github.com/biubug6/Pytorch_Retinaface
# Original license: MIT
import torch
import cv2
import numpy as np
from .. import torch_utils
from .models.retinaface import RetinaFace
from ..box_utils import batched_decode
from .utils import decode_landm
from .config import cfg_re50
from .prior_box import PriorBox
from torch.hub import load_state_dict_from_url


class RetinaNetDetectorONNX(torch.nn.Module):

    def __init__(self, input_imshape, inference_imshape):
        super().__init__()
        self.device = torch.device("cpu")
        cfg = cfg_re50
        state_dict = load_state_dict_from_url(
            "https://folk.ntnu.no/haakohu/RetinaFace_ResNet50.pth",
            map_location=torch_utils.get_device()
        )
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        net = RetinaFace(cfg=cfg)
        net.eval()
        net.load_state_dict(state_dict)
        self.net = net.to(self.device)
        self.input_imshape = input_imshape
        self.inference_imshape = inference_imshape # (height, width)
        self.mean = np.array([104, 117, 123], dtype=np.float32)
        self.mean = torch.from_numpy(self.mean).reshape((1, 3, 1, 1))
        self.mean = torch.nn.Parameter(self.mean).float().to(self.device)
        prior_box = PriorBox(cfg, image_size=self.inference_imshape)
        self.priors = prior_box.forward().to(self.device).data
        self.priors = torch.nn.Parameter(self.priors).float()
        self.variance = torch.nn.Parameter(torch.tensor([0.1, 0.2])).float()

    def export_onnx(self, onnx_filepath):
        try:
            image = cv2.imread("images/0_Parade_marchingband_1_765.jpg")
        except:
            raise FileNotFoundError()
            
        height, width = self.input_imshape
        image = cv2.resize(image, (width, height))

        example_inputs = np.moveaxis(image, -1, 0)
        example_inputs = example_inputs[None]
        np.save("inputs.npy", example_inputs.astype(np.float32))
        example_inputs = torch.from_numpy(example_inputs).float()
        actual_outputs = self.forward(example_inputs).cpu().numpy()

        output_names = ["loc"]
        torch.onnx.export(
            self, example_inputs,
            onnx_filepath,
            verbose=True,
            input_names=["image"],
            output_names=output_names,
            export_params=True,
            opset_version=10 # functional interpolate does not support opset 11+
            )
        np.save(f"outputs.npy", actual_outputs)

    @torch.no_grad()
    def forward(self, image):
        """
            image: shape [1, 3, H, W]
            Exports model where outputs are NOT thresholded or performed NMS on.
        """
        image = torch.nn.functional.interpolate(image, self.inference_imshape, mode="nearest")
        # Expects BGR
        image = image - self.mean
        assert image.shape[2] == self.inference_imshape[0]
        assert image.shape[3] == self.inference_imshape[1]
        assert image.shape[0] == 1,\
            "The ONNX export only supports one image at a time tensors currently"
        loc, conf, landms = self.net(image)  # forward pass
        assert conf.shape[2] == 2
        scores = conf[:, :, 1:]
        boxes = batched_decode(loc, self.priors.data, self.variance, to_XYXY=False)
        landms = decode_landm(landms, self.priors.data, self.variance)
        boxes, landms, scores = [_[0] for _ in [boxes, landms, scores]]
        x0, y0, W, H = [boxes[:, i] for i in range(4)]

        assert boxes.shape[1] == 4
        height, width = image.shape[2:]

        x0 = x0 - W / 2
        y0 = y0 - H / 2
        x1 = x0 + W
        y1 = y0 + H

        boxes = torch.stack((x0, y0, x1, y1), dim=-1)
        return torch.cat((boxes, landms, scores), dim=-1)
