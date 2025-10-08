import cv2
import os

def list_images(folder):
    exts = (".png", ".jpg", ".tif")
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
            if f.lower().endswith(exts)]

def read_gray(path):
    img = cv2.imread(path, 0)
    if img is None:
        raise FileNotFoundError(path)
    return img