import pickle
import os
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
from PIL import Image
from config_paths import IMAGE_DATA_PKL, FEATURE_DATA_PKL
from hashes import get_ahash, get_dhash, get_phash
from histograms import get_histogram
from embeddings import extract_embedding


CHECKPOINT = 2000


def main():
    """
    Load image metadata, compute features (embeddings, histograms, hashes),
    and persist them to a pickle file with periodic checkpoints.
    """
    # Load list of images (id, root, file)
    with open(IMAGE_DATA_PKL, "rb") as f:
        image_data = pickle.load(f)

    feature_data = []

    # If a previous feature dump exists, resume by skipping processed ids
    if os.path.exists(FEATURE_DATA_PKL):
        with open(FEATURE_DATA_PKL, "rb") as f:
            feature_data = pickle.load(f)
        processed_ids = {entry["image_id"] for entry in feature_data}
    else:
        processed_ids = set()

    # Define model for embeddings (pretrained ResNet18 without the
    # classification head)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()
    model.eval().to(device)

    # Iterate through a subset of images (first 5000)
    for images in tqdm(image_data[:5000], desc="Processing images"):
        image_id = images["image_id"]
        filepath = os.path.join(images["root"], images["file"])

        # Skip if already processed
        if image_id in processed_ids:
            continue

        try:
            with Image.open(filepath) as image:
                if image.mode != "RGB":
                    image = image.convert("RGB")

                embeddings = extract_embedding(image, model, device)
                rgb_hists = get_histogram(image)
                ahashes = get_ahash(image)
                dhashes = get_dhash(image)
                phashes = get_phash(image)

            feature_data.append({
                "image_id": image_id,
                "embeddings": embeddings,
                "rgb_hists": rgb_hists,
                "ahashes": ahashes,
                "dhashes": dhashes,
                "phashes": phashes,
            })

        except Exception as e:
            print(f"Error processing image {image_id}: {e}")

        # Periodic checkpoint save
        if len(feature_data) % CHECKPOINT == 0:
            with open(FEATURE_DATA_PKL, "wb") as f:
                pickle.dump(feature_data, f)
            print(
                f"Checkpoint reached. Saved {
                    len(feature_data)} entries to pickle.")

    # Final save
    with open(FEATURE_DATA_PKL, "wb") as f:
        pickle.dump(feature_data, f)
    print(f"Saved remaining {len(feature_data)} entries to pickle.")


if __name__ == "__main__":
    main()
