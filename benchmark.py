import cv2
import time
import face_detection
import tqdm


if __name__ == "__main__":
    num = 1000

    for detector in face_detection.available_detectors:
        detector = face_detection.build_detector(
            detector,
            fp16_inference=True
        )
        im = "images/0_Parade_Parade_0_873.jpg"
        im = cv2.imread(im)[:, :, ::-1]
        t = time.time()
        for i in tqdm.trange(num):
            dets = detector.detect(im)
        total_time = time.time() - t
        avg_time = total_time / num
        fps = 1 / avg_time
        ms = avg_time * 1000
        print(
            f"Detector: {detector.__class__.__name__}. Average inference time over image shape: {im.shape} is:",
            f"{ms:.2f} ms, fps: {fps:.2f}")
