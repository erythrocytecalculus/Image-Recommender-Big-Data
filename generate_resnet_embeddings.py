import os
import gc
import sqlite3
import pickle
import uuid
from typing import List, Dict
from tqdm import tqdm
import cv2
import torch
import numpy as np
from torchvision import models, transforms


# Function to save data as a pickle file
def save_to_pickle(data: Dict, file_path: str):
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


# Function to load data from a pickle file
def load_from_pickle(file_path: str) -> Dict:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return {}


# Function to load checkpoint data from a pickle file
def load_checkpoint(checkpoint_path: str):
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)
    return set()


# Function to save checkpoint data to a pickle file
def save_checkpoint(checkpoint_path: str, processed_uuids: set):
    with open(checkpoint_path, "wb") as f:
        pickle.dump(processed_uuids, f)


# Function to preprocess an image
def preprocess_image(image, preprocess, device):
    image_preprocessed = preprocess(image).unsqueeze(0).to(device)
    return image_preprocessed


# Function to generate embeddings for the images in the database
def generate_embeddings(
    db_path: str, batch_size: int, combined_pickle_path: str, checkpoint_path: str
):
    print("Generating embeddings...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load ResNet50 model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = model.to(device)
    model.eval()

    # Remove the last layer
    model = torch.nn.Sequential(*list(model.children())[:-1])

    # Image preprocessing transformations
    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load processed UUIDs to avoid redundant processing
    processed_uuids = load_checkpoint(checkpoint_path)

    # Connect to the database and fetch image metadata
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT uuid, file_path FROM images")
    rows = cursor.fetchall()

    total_images = len(rows)
    progress_bar = tqdm(total=total_images, desc="Processing Images", unit="image")

    # Dictionary to store combined embeddings
    combined_data = {}

    # Function to process a batch of images
    def process_batch(batch):
        metadata_batch = []
        for image_uuid, file_path in batch:
            if image_uuid in processed_uuids:
                continue
            try:
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError(f"Unable to read image at path {file_path}")

                # Preprocess the image
                image_preprocessed = preprocess_image(image, preprocess, device)

                # Generate embedding using ResNet50 model
                with torch.no_grad():
                    features = model(image_preprocessed)
                    embedding = features.view(features.size(0), -1).cpu().numpy().flatten()

                metadata_batch.append({"uuid": image_uuid, "embedding": embedding})
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")

        return metadata_batch

    try:
        # Process the images in batches
        for start_idx in range(0, total_images, batch_size):
            batch = rows[start_idx : start_idx + batch_size]
            metadata_batch = process_batch(batch)

            # Update combined data and checkpoint
            if metadata_batch:
                combined_data.update(
                    {meta["uuid"]: meta["embedding"] for meta in metadata_batch}
                )
                processed_uuids.update([meta["uuid"] for meta in metadata_batch])
                save_checkpoint(checkpoint_path, processed_uuids)

                progress_bar.update(len(metadata_batch))
                del metadata_batch
                gc.collect()

    except Exception as e:
        print(f"An error occurred: {e}")

    # Save the combined embeddings to pickle file
    save_to_pickle(combined_data, combined_pickle_path)
    progress_bar.close()
    conn.close()


# Main execution
if __name__ == "__main__":
    db_path = "image_metadata.db"  # Database path
    batch_size = 250  # Number of images to process in each batch
    combined_pickle_path = "combined_embeddings.pkl"  # Path to save embeddings
    checkpoint_path = "checkpoint_embeddings.pkl"  # Path to save checkpoint

    # Generate embeddings for the images in the database
    generate_embeddings(db_path, batch_size, combined_pickle_path, checkpoint_path)