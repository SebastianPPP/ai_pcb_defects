import numpy as np
import pandas as pd
from pathlib import Path
import cv2 as cv
from PIL import Image
from numpy import asarray


def read_file_labels(path: Path) -> pd.DataFrame:
    """
    Type:
      0: mouse_bite
      1: spur
      2: missing_hole
      3: short
      4: open_circuit
      5: spurious_copper
    X_max, X_min, Y_max, Y_min:
      - coords of detected objects
    """
    names = ['Type', 'X_max', 'X_min', 'Y_max', 'Y_min', ]
    df = pd.read_csv(path, sep=' ', names=names)
    return df


def read_image_to_matrix(path: Path) -> np.ndarray:
    image = Image.open(path).resize((320, 320))
    return asarray(image) / 255.0

