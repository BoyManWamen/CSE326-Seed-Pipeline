import os

DIRS = [
    "dataset/videos",
    "dataset/images/train",
    "dataset/images/val",
    "dataset/labels_yolo/train",
    "dataset/labels_yolo/val",
    "dataset/meta",
    "docs"
]

for d in DIRS:
    os.makedirs(d, exist_ok=True)

print("Dataset structure initialized.")
