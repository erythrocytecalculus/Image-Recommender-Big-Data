# Image Recommender

## Overview
This repository contains a project designed to identify the top five images most similar to a given photo from a large dataset of nearly 500,000 images. The tool, built in Python, recommends similar images using multiple similarity measures, including color patterns, image embeddings, and YOLO-based object detection.

## Getting Started

### Prerequisites 
Ensure Python is installed on your system and you have the necessary permissions to execute the scripts.

### Installation
Clone the repository to your local machine:
```
https://github.com/erythrocytecalculus/image_recommender.git
```
Navigate to the project directory:
```
cd Image-recommender
```
Install dependencies:
```
pip install . 
```

## 📁 Project Structure

```plaintext
image_recommender/
├── data/                  # Raw image files or paths to image directories
├── db/                    # Database files (e.g. SQLite)
│   └── image_meta.db      # Stores image metadata and file mappings
├── images/                # (Optional) Cached images or thumbnails
├── embeddings/            # Precomputed deep learning feature vectors
├── src/                   # Main source code
│   ├── database.py        # Handles image ID ↔ metadata ↔ file path mappings
│   ├── generator.py       # Image loading and preprocessing pipeline
│   ├── similarity/
│   │   ├── color.py       # Color histogram-based similarity
│   │   ├── embedding.py   # Similarity based on deep features (e.g. CNN)
│   │   ├── custom.py      # Custom similarity metric (e.g. hashing)
│   ├── search.py          # Approximate or brute-force nearest neighbor search
│   ├── recommender.py     # Core logic to combine similarity scores and rank results
│   └── utils.py           # Utility functions (e.g. logging, timers)
├── tests/                 # Unit tests
│   ├── test_similarity.py
│   └── test_database.py
├── requirements.txt       # Python package dependencies
└── main.py                # Entry point to run the recommender (CLI or GUI)
