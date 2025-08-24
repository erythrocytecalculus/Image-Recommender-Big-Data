import os

# Path to the local image dataset
IMAGE_DATA_DIR = r"D:\data\image_data"

# Define project base and subdirectories
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(PROJECT_DIR, "data")
LOG_FOLDER = os.path.join(PROJECT_DIR, "logs")

# Key data file locations
IMAGE_DATA_PKL = os.path.join(DATA_FOLDER, "image_data.pkl")
FEATURE_DATA_PKL = os.path.join(DATA_FOLDER, "feature_data.pkl")
DB_PATH = os.path.join(DATA_FOLDER, "image_database.db")
