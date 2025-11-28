from pathlib import Path
import random

import cv2
from matplotlib import pyplot as plt
from ultralytics import YOLO
from const import DATA_VAL_IMAGES_DIR, YOLO_V8N_160_E8, YOLO_11N_320_E10


def predict_img(
        model_path: Path,
        img_path: Path = None,
        random_val_img: bool = False
):
    if img_path is None and random_val_img:
        paths = [path for path in Path(DATA_VAL_IMAGES_DIR).iterdir()]
        img_path = random.choice(paths)
    model = YOLO(model_path)
    result_ = model(
        img_path,
        imgsz=320,
    )
    return [{
        "xyxy": [res.xyxy for res in r.boxes],
        "plot": r.plot(),
        "confs": [res.conf for res in r.boxes],
        "classes": [res.cls for res in r.boxes]
    } for r in result_][0]


if __name__ == "__main__":
    result = predict_img(YOLO_11N_320_E10, random_val_img=True)
