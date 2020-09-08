# Adapted from https://github.com/biubug6/Pytorch_Retinaface
# Original license: MIT
import torch
import numpy as np
from math import ceil


def generate_prior_box(feature_maps, image_size, steps, min_sizes):
    n_anchors = 0
    for x in feature_maps:
        n_anchors += x[0] * x[1] * len(min_sizes[0])
    anchors = np.empty((n_anchors*4), dtype=np.float32)
#    print(feature_maps, image_size, steps, min_sizes)
    idx_anchor = 0
    for k, f in enumerate(feature_maps):
        min_sizes_ = min_sizes[k]
        for i in range(f[0]):
            for j in range(f[1]):
                for min_size in min_sizes_:
                    s_kx = min_size / image_size[1]
                    s_ky = min_size / image_size[0]
                    dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                    for cy in dense_cy:
                        for cx in dense_cx:
                            anchors[idx_anchor:idx_anchor+4] = [cx, cy, s_kx, s_ky]
                            idx_anchor += 1*4
#    assert idx_anchor == anchors.shape[0], f"{anchors.shape[0]}, {idx_anchor}"
    return anchors


class PriorBox(object):
    def __init__(self, cfg, image_size=None, phase='train'):
        super(PriorBox, self).__init__()
        self.min_sizes =  np.array(cfg['min_sizes']).astype(np.int16)
        self.steps = np.array(cfg['steps']).astype(np.int16)
        self.clip = cfg['clip']
        self.image_size = np.array(image_size).astype(np.int16)
        self.feature_maps = np.array([[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]).astype(np.int16)
        self.name = "s"

    def forward(self):
        anchors = generate_prior_box(
            self.feature_maps, self.image_size, self.steps, self.min_sizes
        )

        # back to torch land
        output = torch.from_numpy(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
