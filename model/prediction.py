from pathlib import Path
import random
from ultralytics import YOLO
from const import DATA_VAL_IMAGES_DIR, YOLO_V8N_160_E8


def predict_img(
        model_path: Path,
        img_path: Path = None,
        random_val_img: bool = False
):
    if img_path is None and random_val_img:
        paths = [path for path in Path(DATA_VAL_IMAGES_DIR).iterdir()]
        img_path = random.choice(paths)
    model = YOLO(model_path)
    model(img_path, save=True, imgsz=320)


if __name__ == "__main__":
    predict_img(YOLO_V8N_160_E8, random_val_img=True)
