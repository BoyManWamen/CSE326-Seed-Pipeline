import argparse
import os
import glob
import cv2
import numpy as np
import pandas as pd

def bucket(n):
    return 3 if n >= 3 else n

def hsv_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brown = cv2.inRange(hsv, (5,30,40), (30,200,220))
    green = cv2.inRange(hsv, (35,60,40), (85,255,255))
    red1  = cv2.inRange(hsv, (0,70,40), (10,255,255))
    red2  = cv2.inRange(hsv, (160,70,40), (179,255,255))

    mask = cv2.bitwise_or(brown, green)
    mask = cv2.bitwise_or(mask, red1)
    mask = cv2.bitwise_or(mask, red2)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask

def contours_to_boxes(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        if 25 <= cv2.contourArea(c) <= 12000:
            x,y,w,h = cv2.boundingRect(c)
            boxes.append((x,y,x+w,y+h))
    return boxes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out_frames_csv", required=True)
    ap.add_argument("--out_boxes_csv", required=True)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    frame_rows = []
    box_rows = []

    for img_path in paths:
        img = cv2.imread(img_path)
        mask = hsv_mask(img)
        boxes = contours_to_boxes(mask)

        frame_rows.append({
            "frame": os.path.basename(img_path),
            "pred_count_raw": len(boxes),
            "pred_count_bucket": bucket(len(boxes))
        })

        for (x1,y1,x2,y2) in boxes:
            box_rows.append({
                "frame": os.path.basename(img_path),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })

    pd.DataFrame(frame_rows).to_csv(args.out_frames_csv, index=False)
    pd.DataFrame(box_rows).to_csv(args.out_boxes_csv, index=False)

    print("Detection complete. CSV files written.")

if __name__ == "__main__":
    main()
