import glob
import os
import cv2
import time
from dsfd import detect_faces


def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)


if __name__ == "__main__":
    impaths = "images"
    impaths = glob.glob(os.path.join(impaths, "*.jpg"))
    confidence_threshold = .3
    for impath in impaths:
        if impath.endswith("out.jpg"): continue
        im = cv2.imread(impath)
        print("Processing:", impath)
        t = time.time()
        dets = detect_faces(
            im, confidence_threshold)[:, :4]
        print(f"Detection time: {time.time()- t:.3f}")
        draw_faces(im, dets)
        imname = os.path.basename(impath).split(".")[0]
        output_path = os.path.join(
            os.path.dirname(impath),
            f"{imname}_out.jpg"
        )

        cv2.imwrite(output_path, im)
        