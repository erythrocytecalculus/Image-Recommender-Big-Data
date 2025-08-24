import cv2
import numpy as np
from PIL import Image


def process_image(image):
    """
    Open an image, ensure RGB mode, and return it.
    """
    img = Image.open(image)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def get_histogram(image, bins=[8, 8, 8]):
    """
    Compute a normalized RGB color histogram and return it as float32.
    """
    arr = np.array(image)
    hist = cv2.calcHist([arr], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()
    return hist.astype(np.float32)


def input_image_histogram(image):
    """
    Wrapper: load/process image and return its histogram.
    """
    img = process_image(image)
    return get_histogram(img)
