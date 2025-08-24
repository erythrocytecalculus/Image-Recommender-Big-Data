import numpy as np
from PIL import Image
import imagehash


def process_image(image):
    """
    Open an image, convert to RGB if needed, and return it.
    """
    img = Image.open(image)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def get_phash(image: Image.Image, hash_size: int = 32) -> np.ndarray:
    """
    Compute a perceptual hash (pHash) using DCT via ImageHash.
    Returns a flattened NumPy array of 0s and 1s.
    """
    ph = imagehash.phash(image, hash_size=hash_size)
    return np.array(ph.hash, dtype=np.uint8).flatten()


def get_ahash(image, hash_size=32):
    """
    Compute an average hash (aHash) for the given image.
    Returns a binary NumPy array.
    """
    img = image.convert("L")
    ah = imagehash.average_hash(img, hash_size=hash_size)
    return ah.hash.astype(np.uint8).flatten()


def get_dhash(image, hash_size=32):
    """
    Compute a difference hash (dHash) for the given image.
    Returns a binary NumPy array.
    """
    img = image.convert("L")
    dh = imagehash.dhash(img, hash_size=hash_size)
    return dh.hash.astype(np.uint8).flatten()


def input_image_ahash(image):
    """
    Wrapper for processing and computing aHash from an input image.
    """
    img = process_image(image)
    return get_ahash(img)


def input_image_dhash(image):
    """
    Wrapper for processing and computing dHash from an input image.
    """
    img = process_image(image)
    return get_dhash(img)


def input_image_phash(image_path: str, hash_size: int = 32) -> np.ndarray:
    """
    Wrapper that opens an image file and computes its pHash as a bit array.
    """
    with Image.open(image_path) as im:
        if im.mode != "RGB":
            im = im.convert("RGB")
        return get_phash(im, hash_size=hash_size)
