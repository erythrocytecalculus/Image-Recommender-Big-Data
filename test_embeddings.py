import pytest
import torch
import numpy as np
from PIL import Image
from io import BytesIO
import sys
import os
from torchvision import models
import torch.nn as nn

# add project src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from embeddings import load_image, prepare_image, extract_embedding, compute_image_embedding


@pytest.fixture
def sample_image():
    """
    Erzeugt ein einfaches 224x224 RGB-Testbild
    """
    img = Image.new("RGB", (224, 224), color="blue")
    return img


def test_process_image(sample_image):
    """
    Prüft, ob das Bild korrekt eingelesen und verarbeitet wird
    """
    # Nutzung von BytesIO, um das sample_image als JPEG im Speicher zu speichern,
    # anstatt eine Datei auf die Festplatte zu schreiben (praktisch für Tests)
    img_bytes = BytesIO()
    sample_image.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    processed_img = load_image(img_bytes)
    assert processed_img.mode == "RGB"
    assert isinstance(processed_img, Image.Image)


def test_preprocess(sample_image):
    """
    Überprüft, ob die Preprocessing-Funktion einen Tensor im richtigen Format zurückgibt
    """
    tensor = prepare_image(sample_image)
    assert isinstance(tensor, torch.Tensor)
    # Erwartet wird eine Batch-Dimension + 3 Farbkanäle + 224x224 Pixel
    assert tensor.shape == (1, 3, 224, 224)


def test_get_embedding(sample_image):
    """
    Testet die Embedding-Berechnung mit ResNet18
    """
    device = torch.device("cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()  # letzte Schicht entfernen
    model.eval()
    model.to(device)

    embedding = extract_embedding(sample_image, model, device)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == 512  # ResNet18 Embedding-Dimension


def test_input_image_embedding(sample_image, monkeypatch):
    """
    Testet die Funktion input_image_embedding mit einem RGB-Testbild
    """
    # Monkeypatching: verhindert erneutes Öffnen des Bildes, nur für Testzwecke
    monkeypatch.setattr("embeddings.process_image", lambda x: x)

    embedding = compute_image_embedding(sample_image)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == 512
