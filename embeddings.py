import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np


def load_image(path):
    """
    Open an image file, ensure it is in RGB format, and return it.
    """
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def prepare_image(image):
    """
    Apply preprocessing steps (resize, crop, normalize) 
    and return a tensor ready for model inference.
    """
    transform_pipeline = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    tensor = transform_pipeline(image)
    return tensor.unsqueeze(0)


def extract_embedding(image, model, device):
    """
    Generate an embedding vector for the given image using a CNN model.
    """
    img_tensor = prepare_image(image).to(device)
    with torch.no_grad():
        vector = model(img_tensor)
    return vector.cpu().numpy().flatten().astype(np.float32)


def compute_image_embedding(image_path):
    """
    High-level function: load image, initialize ResNet18,
    and return the image embedding as a numpy array.
    """
    img = load_image(image_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained ResNet18 with the classification head removed
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval().to(device)

    return extract_embedding(img, model, device)