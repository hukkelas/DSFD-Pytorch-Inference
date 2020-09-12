import torch


def batched_decode(loc, priors, variances, to_XYXY=True):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [N, num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    priors = priors[None]
    boxes = torch.cat((
        priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :,  2:],
        priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1])),
        dim=2)
    if to_XYXY:
        boxes[:, :, :2] -= boxes[:, :, 2:] / 2
        boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


def scale_boxes(imshape, boxes):
    height, width = imshape
    boxes[:, [0, 2]] *= width
    boxes[:, [1, 3]] *= height
    return boxes
