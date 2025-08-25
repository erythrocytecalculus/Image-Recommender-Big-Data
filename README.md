# Image Recommender Big Data

The **Image Recommender System** is a modular framework for recommending images based on **visual similarity**. It allows you to:  

- Generate and store image metadata in a SQL database.  
- Extract features such as **CNN embeddings (ResNet18)**, **RGB color histograms**, and **image hashes (aHash, dHash, pHash)**.  
- Query the system with an input image and retrieve the *top-k* most similar results using metrics like **cosine similarity, ANN cosine, Euclidean, Manhattan, Hamming, and Bhattacharyya distance**.  
- Visualize feature distributions with TensorBoard or custom plots.  

## Table of Contents  

- [Overview](#overview)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Visualization](#visualization)  

---

## Overview  

The pipeline consists of four main components:  

1. **Metadata & Database Setup**  
   - Scans the dataset and extracts metadata (IDs, file paths).  
   - Stores this data in a SQL database for indexing and querying.  

2. **Feature Extraction**  
   - **Embeddings**: ResNet18 generates semantic feature vectors.  
   - **RGB Histograms**: Encodes the global distribution of colors in an image.  
   - **Hashing**: aHash, dHash, and pHash for structural and perceptual similarity.  

3. **Similarity Computation**  
   - Supported distance functions:  
     - **Cosine**  
     - **ANN Cosine** (Approximate Nearest Neighbor search)  
     - **Euclidean**  
     - **Manhattan**  
     - **Hamming**  
     - **Bhattacharyya distance**  
   - Returns the *top-k* most similar database images.  

4. **Exploration & Visualization**  
   - Jupyter notebook (`show_similarites.ipynb`) for interactive similarity search.  
   - TensorBoard and dimensionality reduction (e.g., UMAP) for embedding visualization.  


## Installation  

1. **Clone this repository**  
```bash
git clone https://github.com/erythrocytecalculus/Image-Recommender-Big-Data.git
```

2. **Navigate to the project directory**
```
cd image_recommender
```

3. **Install the dependencies**
```
pip install -r requirements.txt
```

---

## Usage

After setting up the repository, the pipeline runs in several stages:

**1. Configure your dataset**  
- Open the configuration file (`config.py`).  
- Replace the example path with the location of your own dataset, e.g.:  

  ```python
  # Path to the local image dataset
  IMAGE_DATA_DIR = r"D:\data\image_data"


2. **Run the generator to build the database**
Execute the following command from the project’s root directory:  
```
python generator.py
```

3. **Run the main pipeline to extract features**

```
python main.py
```

**4. View similarity results**  
- Launch the Jupyter Notebook `show_similarites.ipynb`.  
- Insert the file paths of your query images into the input list.  
- The notebook will call the `calculate_similarites()` function from `similarites.py`, which supports the following arguments:  

  - **input_images**: list of query image paths  
  - **cursor**: database cursor for accessing stored features  
  - **mode**: similarity method (`embeddings | rgb | ahash | dhash | phash`)  
  - **metric**: distance measure (`cosine | ann_cosine | euclidean | manhattan | hamming | bhattacharyya`)  
  - **top_k**: number of nearest results to return (default = 5)  
  - **verbose**: set to `True` to log runtime details (default = `False`)  


---


## **Visualisation**

You can visualize the images either in a Tensorboard or with Dimensionreduction
