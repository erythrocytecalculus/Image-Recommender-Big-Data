import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from hashes import (
    process_image,
    get_ahash,
    get_dhash,
    get_phash,              # NEW
    input_image_ahash,
    input_image_dhash,
    input_image_phash,      # NEW
)


@pytest.fixture
def create_dummy_image(tmp_path):
    """
    Creates a dummy image for the tests.
    """
    image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(image_data)

    image_path = tmp_path / "dummy_image.jpg"
    image.save(image_path)

    return image_path


def test_process_image(create_dummy_image):
    """
    Tests the processing of a single image.
    """
    image_path = create_dummy_image
    processed_image = process_image(image_path)

    assert isinstance(processed_image, Image.Image)
    assert processed_image.size == (100, 100)
    assert processed_image.mode == "RGB"


def test_get_ahash(create_dummy_image):
    """
    Tests computation of the average hash (aHash).
    """
    image_path = create_dummy_image
    image = process_image(image_path)

    hash_array = get_ahash(image)

    assert isinstance(hash_array, np.ndarray)
    assert hash_array.shape == (32 * 32,)
    assert np.all(np.isin(hash_array, [0, 1]))


def test_get_dhash(create_dummy_image):
    """
    Tests computation of the difference hash (dHash).
    """
    image_path = create_dummy_image
    image = process_image(image_path)

    hash_array = get_dhash(image)

    assert isinstance(hash_array, np.ndarray)
    assert hash_array.shape == (32 * 32,)
    assert np.all(np.isin(hash_array, [0, 1]))

def test_get_phash(create_dummy_image):
    """
    Tests computation of the perceptual hash (pHash) given a PIL image.
    """
    image_path = create_dummy_image
    image = process_image(image_path)

    hash_array = get_phash(image)  # default hash_size=32

    assert isinstance(hash_array, np.ndarray)
    assert hash_array.shape == (32 * 32,)
    assert np.all(np.isin(hash_array, [0, 1]))


def test_input_image_phash(create_dummy_image):
    """
    Tests the input-image pHash wrapper (path -> bit array).
    """
    image_path = create_dummy_image

    hash_array = input_image_phash(str(image_path))  # default hash_size=32

    assert isinstance(hash_array, np.ndarray)
    assert hash_array.shape == (32 * 32,)
    assert np.all(np.isin(hash_array, [0, 1]))
