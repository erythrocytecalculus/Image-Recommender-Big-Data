import pickle
import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from config_paths import IMAGE_DATA_PKL, FEATURE_DATA_PKL, LOG_FOLDER
from tensorboard.plugins import projector
import tensorflow as tf


# Create a single sprite image from many smaller images
def create_sprite(data):
    """
    Tile a batch of images into one sprite image.
    Pads to the next perfect square if needed.

    Args:
        data (np.ndarray): Shape (num_images, H, W, C) or (num_images, H, W)

    Returns:
        np.ndarray: Sprite image of shape (N*H, N*W, 3) where N is grid side.
    """
    # If grayscale, broadcast to 3 channels
    if len(data.shape) == 3:  # (N, H, W)
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))  # -> (N, H, W, 3)

    num_images = data.shape[0]
    n = int(np.ceil(np.sqrt(num_images)))  # grid size N x N

    # Pad up to N^2 images
    padding = ((0, n**2 - num_images), (0, 0), (0, 0), (0, 0))
    data = np.pad(data, padding, mode="constant", constant_values=0)

    # Arrange into grid and collapse
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3, 4))
    data = data.reshape((n * data.shape[1], n * data.shape[3], 3))
    return data


def create_metadata_file(image_data, metadata_path):
    """
    Write one line per image to the metadata file (here: image_id only).

    Args:
        image_data (list[dict]): Entries with at least 'image_id'.
        metadata_path (str): Output TSV path.
    """
    with open(metadata_path, "w") as f:
        for entry in image_data:
            f.write(f"{entry['image_id']}\n")


def load_image_data(image_data_path):
    with open(image_data_path, "rb") as f:
        return pickle.load(f)


def process_images(image_paths, image_size=(12, 12)):
    """
    Load, RGB-convert (if needed), resize, and stack images into an array.
    """
    image_arrays = []
    for path in tqdm(image_paths):
        try:
            img = Image.open(path).resize(image_size)
            if img.mode != "RGB":
                img = img.convert("RGB")
            image_arrays.append(np.array(img))
        except Exception as e:
            print(f"Skipping {path}: {e}")
    return np.stack(image_arrays)


def prepare_tensorboard_data(mode):
    """
    Create TensorBoard Projector assets:
      - checkpoint with the selected feature matrix
      - projector config
      - references to metadata and sprite
    """
    # Clean previous checkpoints for a fresh run
    for filename in os.listdir(LOG_FOLDER):
        if filename.startswith("embedding.ckpt") or filename == "checkpoint":
            os.remove(os.path.join(LOG_FOLDER, filename))

    # Load features and select the requested mode (e.g., 'embeddings', 'rgb_hists')
    with open(FEATURE_DATA_PKL, "rb") as f:
        embeddings_data = pickle.load(f)
    embeddings_array = np.array([entry[mode] for entry in embeddings_data])

    # Create a TF variable and save a checkpoint
    embedding_var = tf.Variable(embeddings_array, name="embedding")
    checkpoint = tf.train.Checkpoint(embedding=embedding_var)
    checkpoint.save(os.path.join(LOG_FOLDER, "embedding.ckpt"))

    # Configure projector
    config = projector.ProjectorConfig()
    emb = config.embeddings.add()
    # Tensor is saved under this internal name by TF
    emb.tensor_name = "embedding/.ATTRIBUTES/VARIABLE_VALUE"
    emb.metadata_path = "metadata.tsv"

    # Sprite settings
    emb.sprite.image_path = "sprite.png"
    emb.sprite.single_image_dim.extend([12, 12])

    projector.visualize_embeddings(LOG_FOLDER, config)
    print(f"Projector config saved at: {LOG_FOLDER}")


def main(mode):
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Load image metadata
    image_data = load_image_data(IMAGE_DATA_PKL)[:5000]

    # Create metadata file (once)
    metadata_path = os.path.join(LOG_FOLDER, "metadata.tsv")
    if not os.path.exists(metadata_path):
        create_metadata_file(image_data, metadata_path)

    # Build list of image paths (by id)
    image_id_to_path = {
        entry["image_id"]: os.path.join(entry["root"], entry["file"]) for entry in image_data
    }
    image_paths = [image_id_to_path[entry["image_id"]] for entry in image_data]

    # Create sprite image (once)
    sprite_image_path = os.path.join(LOG_FOLDER, "sprite.png")
    if not os.path.exists(sprite_image_path):
        image_data_array = process_images(image_paths)
        sprite_image = create_sprite(image_data_array)
        Image.fromarray(sprite_image.astype(np.uint8)).save(sprite_image_path)
        print("Sprite image saved as logs/sprite.png")

    # Build projector assets for the selected feature mode
    prepare_tensorboard_data(mode)


if __name__ == "__main__":
    # main("embeddings")
    main("rgb_hists")
    # Run in terminal:
    # tensorboard --logdir logs/
