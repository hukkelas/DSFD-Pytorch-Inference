import glob
import os
import cv2
import time
import argparse
import logging

import face_detection


def draw_faces(im, bboxes):
    for bbox in bboxes:
        x0, y0, x1, y1 = [int(_) for _ in bbox]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 0, 255), 2)


if __name__ == "__main__":
    
    logging.basicConfig(level=getattr(logging, "INFO"))

    parser = argparse.ArgumentParser(
        prog="DSDF face detector",
        description="Face detector based on AI"
    )
    parser.add_argument("--img_path", type=str, required=True, 
                        help="path to single image or a folder where many images are stored")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["DSFDDetector", "RetinaNetResNet50", "RetinaNetMobileNetV1"],
                        default="DSFDDetector",
                        help="Model to use")
    parser.add_argument("--model_weights", type=str, required=True, 
                        help="Path to the downloaded model weights")
    parser.add_argument("--confidence_threshold", type=float, default=0.3)
    parser.add_argument("--nms_iou_threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_resolution", type=int, default=1080)
    parser.add_argument("--fp16_inference", type=bool, default=True)
    parser.add_argument("--clip_boxes", type=bool, default=False)
    parser.add_argument("--out_folder", type=str, default="Folder where the output images will be saved")

    args = parser.parse_args()

    if os.path.isfile(args.img_path):
        logging.info("Single image detected")
        impaths = [args.img_path,] 
    else:
        impaths = glob.glob(os.path.join(args.img_path, "*"))
        logging.info(f"Many images detected (total={len(impaths)})")

    detector = face_detection.build_detector(
        name=args.model,
        confidence_threshold=args.confidence_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
        device=args.device,
        max_resolution=args.max_resolution,
        fp16_inference=args.fp16_inference,
        clip_boxes=args.clip_boxes,
        model_weights=args.model_weights,
    )
    logging.info(f"Model {args.model} loaded with weights {args.model_weights}")

    if not os.path.isdir(args.out_folder):
        os.makedirs(args.out_folder)

    for impath in impaths:
        im = cv2.imread(impath)
        logging.info(f"Processing: {impath}")
        t = time.time()
        dets = detector.detect(
            im[:, :, ::-1]
        )[:, :4]
        logging.info(f"Detection time: {time.time()- t:.3f}")
        draw_faces(im, dets)
        
        imname = os.path.basename(impath).split(".")[0]
        output_path = os.path.join(args.out_folder,f"{imname}.jpg")

        cv2.imwrite(output_path, im)
        