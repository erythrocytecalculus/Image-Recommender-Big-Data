import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
from config_paths import FEATURE_DATA_PKL
from sklearn.metrics.pairwise import cosine_similarity
from database_utils import fetch_image_path
from histograms import input_image_histogram
from hashes import input_image_ahash, input_image_dhash, input_image_phash
from embeddings import compute_image_embedding

# --- FAISS (optional; graceful fallback provided) ---
try:
    import faiss  # pip install faiss-cpu
except Exception:
    faiss = None

# Reusable globals so we build the FAISS index only once per session
FAISS_INDEX = None
FAISS_IDS = None
FAISS_MATRIX = None

FEATURE_DATA_CACHE = None


def plot_similar_images(input_images, top_similarities,
                        target_size=(512, 512)):
    """
    Visualize the query image(s) alongside the top-k similar results.

    Args:
        input_images (list[str]): Paths to the query image files.
        top_similarities (list[tuple]): (image_id, score, filepath) for retrieved items.
        target_size (tuple): (width, height) to render all images uniformly.
    """
    # Build a horizontal strip of all query images
    combined_width = target_size[0] * len(input_images)
    combined_image = Image.new("RGB", (combined_width, target_size[1]))
    for i, image in enumerate(input_images):
        img = Image.open(image).resize(target_size).convert("RGB")
        combined_image.paste(img, (i * target_size[0], 0))

    # Create figure: [query strip] | divider | [results...]
    fig, axs = plt.subplots(
        1,
        len(top_similarities) + 2,
        figsize=(15, 5),
        gridspec_kw={
            "width_ratios": [
                1.5,
                0.01] +
            [1] *
            len(top_similarities)},
    )
    axs[0].imshow(combined_image)
    axs[0].axis("off")
    axs[0].set_title("Input Image")

    axs[1].axis("off")
    axs[1].add_patch(patches.Rectangle((0, 0), 1, 1, color="black"))

    for i, (_, sim_score, filepath) in enumerate(top_similarities):
        img = Image.open(filepath).resize(target_size)
        axs[i + 2].imshow(img)
        axs[i + 2].axis("off")
        axs[i + 2].set_title(f"Sim_score: {sim_score:.2f}")

    plt.tight_layout()
    plt.show()


def load_feature_data():
    """Load feature rows from disk once and cache them in memory."""
    global FEATURE_DATA_CACHE
    if FEATURE_DATA_CACHE is not None:
        return FEATURE_DATA_CACHE
    with open(FEATURE_DATA_PKL, "rb") as f:
        FEATURE_DATA_CACHE = pickle.load(f)
    return FEATURE_DATA_CACHE


def _bhattacharyya_distance(h1, h2, eps: float = 1e-12) -> float:
    """
    Compute Bhattacharyya distance between two histograms (smaller => more similar).

    Steps:
      1) L1-normalize both histograms.
      2) Compute coefficient BC = sum(sqrt(h1 * h2)).
      3) Distance = -ln(BC + eps).
    """
    h1 = h1.astype(float)
    h2 = h2.astype(float)
    s1, s2 = h1.sum(), h2.sum()
    if s1 > 0:
        h1 = h1 / s1
    if s2 > 0:
        h2 = h2 / s2
    bc = np.sum(np.sqrt(np.clip(h1, 0, None) * np.clip(h2, 0, None)))
    return float(-np.log(bc + eps))


def _ensure_faiss_index(data):
    """
    Build a cosine-similarity FAISS index (IndexFlatIP) over embeddings once.

    Returns:
        bool: True if FAISS is available and the index is ready; False otherwise.
    """
    global FAISS_INDEX, FAISS_IDS, FAISS_MATRIX, faiss
    if faiss is None:
        return False
    if FAISS_INDEX is not None:
        return True

    # Stack embeddings in the same order as `data`
    X = np.vstack([np.asarray(entry["embeddings"], dtype=np.float32)
                  for entry in data])
    FAISS_IDS = np.array([entry["image_id"] for entry in data], dtype=np.int64)

    # L2-normalize so cosine ~ inner product
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    FAISS_MATRIX = X / norms

    # Flat inner-product index (cosine ≈ IP on normalized vectors)
    FAISS_INDEX = faiss.IndexFlatIP(FAISS_MATRIX.shape[1])
    FAISS_INDEX.add(FAISS_MATRIX)
    return True


def calculate_similarities(input_images, cursor, mode,
                           metric, top_k=5, verbose=False):
    """
    Compute top-k similar images under a chosen feature `mode` and `metric`.

    Args:
        input_images (list[str]): Paths to query images.
        cursor: SQLite cursor used by select_image_from_database.
        mode (str): One of {'rgb','embeddings','ahashes','dhashes','phashes'}.
        metric (str): One of {'cosine','euclidean','manhattan','hamming','bhattacharyya','ann_cosine'}.
        top_k (int): Number of nearest results to return.
        verbose (bool): If True, print timing breakdowns.

    Returns:
        list[tuple]: (image_id, similarity_or_distance, filepath) for the top-k results.
    """
    global FEATURE_DATA_CACHE
    mode = mode.lower()
    metric = metric.lower()
    start_time = time.time()

    # Load features
    t0 = time.time()
    data = load_feature_data()
    t1 = time.time()
    if verbose:
        print(f"Time to load feature data: {t1 - t0:.4f} s")

    # ---- Feature extraction for the query ----
    t2 = time.time()
    if mode == "rgb":
        input_transformed = np.mean(
            [input_image_histogram(image) for image in input_images], axis=0
        )
        X = np.array([entry["rgb_hists"] for entry in data])

    elif mode == "embeddings":
        input_transformed = np.mean(
            [compute_image_embedding(image) for image in input_images], axis=0
        )
        X = np.array([entry["embeddings"] for entry in data])

    elif mode == "ahashes":
        input_transformed = np.mean(
            [input_image_ahash(image) for image in input_images], axis=0
        )
        X = np.array([entry["ahashes"] for entry in data])

    elif mode == "dhashes":
        input_transformed = np.mean(
            [input_image_dhash(image) for image in input_images], axis=0
        )
        X = np.array([entry["dhashes"] for entry in data])

    elif mode == "phashes":
        input_transformed = np.mean(
            [input_image_phash(image) for image in input_images], axis=0
        )
        X = np.array([entry["phashes"] for entry in data])

    else:
        raise ValueError(
            "Invalid mode. Choose from 'rgb', 'embeddings', 'ahashes', 'phashes', 'dhashes'."
        )

    # For hash modes, re-binarize after averaging multiple queries
    if mode in {"ahashes", "dhashes", "phashes"}:
        input_transformed = (input_transformed > 0.5).astype(np.uint8)
        X = (X > 0.5).astype(np.uint8)

    t3 = time.time()
    if verbose:
        print(f"Time to extract features: {t3 - t2:.4f} s")

    # ---- Similarity / distance computation ----
    t4 = time.time()

    if metric == "cosine":
        input_vec = input_transformed.reshape(1, -1)
        sims = cosine_similarity(input_vec, X).flatten()  # higher = better
        top_k_indices = np.argpartition(-sims, top_k)[:top_k]
        top_k_indices = top_k_indices[np.argsort(-sims[top_k_indices])]

    elif metric == "euclidean":
        sims = np.linalg.norm(
            X - input_transformed,
            axis=1)  # smaller = better
        top_k_indices = np.argpartition(sims, top_k)[:top_k]
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])]

    elif metric == "manhattan":
        sims = np.sum(
            np.abs(
                X - input_transformed),
            axis=1)  # smaller = better
        top_k_indices = np.argpartition(sims, top_k)[:top_k]
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])]

    elif metric == "hamming":
        sims = np.sum(input_transformed != X, axis=1)  # smaller = better
        top_k_indices = np.argpartition(sims, top_k)[:top_k]
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])]

    elif metric == "bhattacharyya":
        if mode != "rgb":
            raise ValueError("Bhattacharyya is only valid with mode='rgb'.")
        sims = np.array(
            [_bhattacharyya_distance(input_transformed, row) for row in X],
            dtype=float,
        )  # smaller = better
        top_k_indices = np.argpartition(sims, top_k)[:top_k]
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])]

    elif metric == "ann_cosine":
        if mode != "embeddings":
            raise ValueError(
                "ann_cosine is only valid with mode='embeddings'.")
        has_faiss = _ensure_faiss_index(data)

        # Normalize query vector
        q = np.asarray(input_transformed, dtype=np.float32)
        q /= (np.linalg.norm(q) + 1e-12)

        if has_faiss:
            # cosine via IP on normed vecs
            D, I = FAISS_INDEX.search(q[None, :], top_k)
            sims = D[0]  # higher = better
            top_k_indices = I[0]
        else:
            # Fallback: NumPy-based cosine similarity
            Xemb = np.array([entry["embeddings"]
                            for entry in data], dtype=np.float32)
            Xn = Xemb / (np.linalg.norm(Xemb, axis=1, keepdims=True) + 1e-12)
            sims = (Xn @ q).astype(np.float32)  # higher = better
            top_k_indices = np.argpartition(-sims, top_k)[:top_k]
            top_k_indices = top_k_indices[np.argsort(-sims[top_k_indices])]

    else:
        raise ValueError(
            "Invalid metric! Use 'cosine', 'euclidean', 'manhattan', 'hamming', 'bhattacharyya', or 'ann_cosine'."
        )

    t5 = time.time()
    if verbose:
        print(f"Time to compute similarities: {t5 - t4:.4f} s")

    # ---- Retrieve top-k image paths from DB ----
    t6 = time.time()
    top_similarities = []
    if len(sims) == len(data):
        # Non-ANN paths: one score per database row
        for i in top_k_indices:
            image_id = data[i]["image_id"]
            sim_value = sims[i]
            file_path = fetch_image_path(image_id, cursor)
            if file_path:
                top_similarities.append((image_id, sim_value, file_path))
    else:
        # ANN path: sims contains only the top_k scores
        for rank, i in enumerate(top_k_indices):
            image_id = data[i]["image_id"]
            sim_value = sims[rank]
            file_path = fetch_image_path(image_id, cursor)
            if file_path:
                top_similarities.append((image_id, sim_value, file_path))
    t7 = time.time()
    if verbose:
        print(f"Time to retrieve top-K images: {t7 - t6:.4f} s")

    total_time = time.time() - start_time
    if verbose:
        print(f"Total execution time: {total_time:.4f} s")

    # ---- Plot results ----
    t8 = time.time()
    plot_similar_images(input_images, top_similarities)
    t9 = time.time()
    if verbose:
        print(f"Time to plot images: {t9 - t8:.4f} s")
