import numpy as np
import torch
import os


def image_to_torch(image, device):
    if image.dtype == np.uint8:
        image = image.astype(np.float32)
    else:
        assert image.dtype == np.float32
    image = np.rollaxis(image, 2)
    image = image[None, :, :, :]
    image = torch.from_numpy(image).to(device)
    return image


def load_weights(weights_path):
    if os.path.isfile(weights_path):
        state_dict = torch.load(weights_path, map_location="cpu")
    else:
        state_dict = torch.hub.load_state_dict_from_url(
                weights_path,
                map_location="cpu"
            )
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    return state_dict
