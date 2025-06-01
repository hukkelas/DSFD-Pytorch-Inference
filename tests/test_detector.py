import numpy as np
import pytest
import cv2
import face_detection  # your face detection library

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    print(boxA, boxB)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        print("Ret 0")
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)

    print("IoU", iou)

    return iou


@pytest.fixture
def ground_truth_boxes():
    return np.array([
        [337.8219142,  227.30235955, 363.18236876, 260.75754449],
        [120.61462998, 244.68149829, 153.73102021, 290.13813281],
        [793.31824303,  88.6468603,  837.80744743, 153.03655452],
        [499.23486614, 212.40574998, 521.46317768, 241.84556359],
        [412.37690353, 219.29100847, 437.20971298, 250.56026506],
        [654.66749144, 203.24960518, 676.66251707, 231.10678673],
        [692.63414764, 248.56575656, 726.75259781, 292.49138522],
        [215.16035197, 269.50566196, 240.76163981, 303.02491093],
        [189.08402371, 212.22481942, 210.5982945,  240.76419282],
        [571.04836243, 213.0569253,  590.01044816, 238.5836339],
        [ 16.7418344,  235.77498758,  41.44155097, 265.93795145],
        [284.28320718, 213.93544269, 304.40658212, 238.0858829],
        [167.58154631,  76.92867303, 187.13439512, 102.97041345],
    ])

@pytest.mark.parametrize("detector_name", [
    "DSFDDetector",
    "RetinaNetResNet50",
    "RetinaNetMobileNetV1"
])
def test_detector_detects_boxes_with_iou(detector_name,  ground_truth_boxes):
    detector = face_detection.build_detector(
        detector_name,
        max_resolution=1080,
        confidence_threshold=0.5
    )
    impath = "images/11_Meeting_Meeting_11_Meeting_Meeting_11_176.jpg"
    img = cv2.imread(impath)

    detections = detector.detect(img[:, :, ::-1])[:, :4]

    for gt_box in ground_truth_boxes:
        print("CHECKING")
        matched = any(compute_iou(gt_box, det_box) >= 0.5 for det_box in detections)
        assert matched, (
            f"{detector_name} failed to detect ground truth box {gt_box} "
            f"with IoU >= 0.5"
        )

