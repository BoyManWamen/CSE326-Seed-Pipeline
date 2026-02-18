import cv2
import numpy as np
import pandas as pd
import random
import math
import os
from dataclasses import dataclass

@dataclass
class Seed:
    x: float
    y: float
    vx: float
    vy: float
    ax1: float
    ax2: float
    angle: float
    kind: str

def seed_color_bgr(kind: str) -> tuple[int, int, int]:
    if kind == "brown":
        return (random.randint(10, 40), random.randint(30, 90), random.randint(70, 150))
    if kind == "green":
        return (random.randint(20, 60), random.randint(140, 230), random.randint(20, 80))
    if kind == "red":
        return (random.randint(20, 70), random.randint(20, 70), random.randint(150, 240))
    return (128, 128, 128)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def ellipse_bbox(seed: Seed, w: int, h: int):
    theta = math.radians(seed.angle)
    cos_t, sin_t = abs(math.cos(theta)), abs(math.sin(theta))
    bw = 2 * (seed.ax1 * cos_t + seed.ax2 * sin_t)
    bh = 2 * (seed.ax1 * sin_t + seed.ax2 * cos_t)
    x1 = seed.x - bw / 2
    y1 = seed.y - bh / 2
    x2 = seed.x + bw / 2
    y2 = seed.y + bh / 2
    x1 = clamp(x1, 0, w - 1); y1 = clamp(y1, 0, h - 1)
    x2 = clamp(x2, 0, w - 1); y2 = clamp(y2, 0, h - 1)
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    return x1, y1, x2, y2

def make_background(h: int, w: int) -> np.ndarray:
    base = random.randint(25, 60)
    img = np.full((h, w, 3), base, dtype=np.uint8)
    x1, x2 = int(w * 0.35), int(w * 0.65)
    img[:, x1:x2] = (base - 10)
    noise = np.random.normal(0, random.uniform(2, 8), (h, w, 3)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    for _ in range(random.randint(2, 6)):
        p1 = (random.randint(0, w - 1), random.randint(0, h - 1))
        p2 = (clamp(p1[0] + random.randint(-30, 30), 0, w - 1),
              clamp(p1[1] + random.randint(50, 140), 0, h - 1))
        cv2.line(img, p1, p2, (base + random.randint(5, 25),) * 3, thickness=random.randint(1, 2))
    return img

def spawn_seed(w: int, h: int) -> Seed:
    x = random.uniform(w * 0.38, w * 0.62)
    y = random.uniform(-40, -10)
    vx = random.uniform(-0.3, 0.3)
    vy = random.uniform(2.0, 5.5)
    ax1 = random.uniform(6, 12)
    ax2 = random.uniform(3, 7)
    angle = random.uniform(0, 180)
    kind = random.choices(["brown", "green", "red"], weights=[0.45, 0.3, 0.25])[0]
    return Seed(x, y, vx, vy, ax1, ax2, angle, kind)

def apply_effects(img: np.ndarray) -> np.ndarray:
    if random.random() < 0.8:
        alpha = random.uniform(0.85, 1.25)
        beta = random.uniform(-18, 18)
        img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if random.random() < 0.6:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), sigmaX=random.uniform(0.2, 1.1))
    if random.random() < 0.12:
        k = random.choice([7, 9, 11])
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        img = cv2.filter2D(img, -1, kernel)
    return img

def main():
    random.seed(42)
    np.random.seed(42)

    # ✅ SAVE NEXT TO THIS .PY FILE (guaranteed)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    out_video = os.path.join(BASE_DIR, "falling_seeds.mp4")
    out_gt_csv = os.path.join(BASE_DIR, "ground_truth_frames.csv")
    out_boxes_csv = os.path.join(BASE_DIR, "ground_truth_boxes.csv")

    w, h = 640, 480
    fps = 30
    seconds = 12
    total_frames = fps * seconds

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (w, h))

    seeds: list[Seed] = []
    gt_rows = []
    spawn_prob = 0.12

    for t in range(total_frames):
        frame = make_background(h, w)

        if random.random() < spawn_prob:
            seeds.append(spawn_seed(w, h))
        if random.random() < 0.03:
            for _ in range(random.randint(1, 3)):
                seeds.append(spawn_seed(w, h))

        boxes = []
        keep = []
        for s in seeds:
            s.vy += random.uniform(0.0, 0.05)
            s.x += s.vx + random.uniform(-0.25, 0.25)
            s.y += s.vy
            s.angle = (s.angle + random.uniform(-6, 6)) % 180
            s.x = clamp(s.x, w * 0.33 + 10, w * 0.67 - 10)

            color = seed_color_bgr(s.kind)
            cv2.ellipse(frame, (int(s.x), int(s.y)), (int(s.ax1), int(s.ax2)),
                        s.angle, 0, 360, color, -1, lineType=cv2.LINE_AA)

            hl = (min(255, color[0] + 20), min(255, color[1] + 20), min(255, color[2] + 20))
            cv2.ellipse(frame, (int(s.x - 0.2 * s.ax1), int(s.y - 0.2 * s.ax2)),
                        (max(1, int(s.ax1 * 0.5)), max(1, int(s.ax2 * 0.5))),
                        s.angle, 0, 360, hl, -1, lineType=cv2.LINE_AA)

            x1, y1, x2, y2 = ellipse_bbox(s, w, h)
            boxes.append((x1, y1, x2, y2, s.kind))

            if s.y < h + 60:
                keep.append(s)
        seeds = keep

        frame = apply_effects(frame)
        writer.write(frame)

        gt_rows.append({
            "frame_idx": t,
            "time_s": t / fps,
            "true_count_raw": len(boxes),
            "true_count_bucket": 3 if len(boxes) >= 3 else len(boxes),
            "boxes": boxes
        })

    writer.release()

    df_frames = pd.DataFrame([{
        "frame_idx": r["frame_idx"],
        "time_s": r["time_s"],
        "true_count_raw": r["true_count_raw"],
        "true_count_bucket": r["true_count_bucket"],
    } for r in gt_rows])
    df_frames.to_csv(out_gt_csv, index=False)

    box_rows = []
    for r in gt_rows:
        for (x1, y1, x2, y2, kind) in r["boxes"]:
            box_rows.append({
                "frame_idx": r["frame_idx"],
                "time_s": r["time_s"],
                "kind": kind,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2
            })
    pd.DataFrame(box_rows).to_csv(out_boxes_csv, index=False)

    print("Wrote:", out_video)
    print("Wrote:", out_gt_csv)
    print("Wrote:", out_boxes_csv)

if __name__ == "__main__":
    main()
