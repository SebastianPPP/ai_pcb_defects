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

    model = YOLO("model/v11n10e320.pt")
    result_ = model(
        "data/test/images/l_light_01_missing_hole_07_2_600.jpg",
        imgsz=320
    )
    print(result_[0])
    return [{
        "xyxy": [res.xyxy for res in r.boxes],
        "plot": r.plot(),
        "confs": [res.conf for res in r.boxes],
        "classes": [res.cls for res in r.boxes]
    } for r in result_][0]


if __name__ == "__main__":
    result = predict_img(YOLO_11N_320_E10, random_val_img=True)
