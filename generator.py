import os
import pickle
from tqdm import tqdm
from config_paths import IMAGE_DATA_DIR, DB_PATH, IMAGE_DATA_PKL
from database_utils import build_database


# Generator that iterates through folders and yields (root, filename)
def generator_to_pickle(path_to_dir):
    for root, _, files in os.walk(path_to_dir):
        for file in files:
            if file.endswith((".png", ".jpg", ".jpeg")):
                # size = os.path.getsize(os.path.join(root, file))  # optional
                yield root, file


# Store generator output into a pickle file
def save_data_to_pickle(gen, output_path):
    image_id = 0
    image_metadata_list = []
    for root, file in tqdm(gen, desc="Processing"):
        record = {"image_id": image_id, "root": root, "file": file}
        image_metadata_list.append(record)
        image_id += 1

    with open(output_path, "wb") as f:
        pickle.dump(image_metadata_list, f)


def main():
    gen = generator_to_pickle(IMAGE_DATA_DIR)
    save_data_to_pickle(gen, IMAGE_DATA_PKL)
    build_database(DB_PATH, IMAGE_DATA_PKL)


if __name__ == "__main__":
    main()