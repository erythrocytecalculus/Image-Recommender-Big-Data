import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from PIL import Image
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from histograms import process_image, get_histogram, input_image_histogram


@pytest.fixture
def create_dummy_image(tmp_path):
    """
    Creates a dummy image for tests.
    """
    image_data = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    image = Image.fromarray(image_data)
    image_path = tmp_path / "dummy_image.jpg"
    image.save(image_path)
    return image_path


def test_process_image(create_dummy_image):
    """
    Tests processing of a single image.
    """
    image_path = create_dummy_image
    processed_image = process_image(image_path)

    assert isinstance(processed_image, Image.Image)
    assert processed_image.size == (100, 100)
    assert processed_image.mode == "RGB"


def test_get_histogram(create_dummy_image):
    """
    Tests computation of the color histogram.
    """
    image_path = create_dummy_image
    image = process_image(image_path)

    histogram = get_histogram(image)

    assert isinstance(histogram, np.ndarray)
    assert histogram.shape == (512,)          # 8*8*8 RGB bins
    assert np.isclose(histogram.sum(), 1.0)   # L1-normalized


def test_input_image_histogram(create_dummy_image):
    """
    Tests the input-image histogram wrapper.
    """
    image_path = create_dummy_image

    histogram = input_image_histogram(image_path)

    assert isinstance(histogram, np.ndarray)
    assert histogram.shape == (512,)
    assert np.isclose(histogram.sum(), 1.0)