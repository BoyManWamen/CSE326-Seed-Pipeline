import argparse
import os
import cv2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every_n", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cap = cv2.VideoCapture(args.video)

    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.every_n == 0:
            path = os.path.join(args.out, f"frame_{idx:06d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1
        idx += 1

    cap.release()
    print(f"Saved {saved} frames")

if __name__ == "__main__":
    main()
