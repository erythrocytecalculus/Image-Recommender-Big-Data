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

# Globals
FAISS_INDEX = None
FAISS_IDS = None
FAISS_MATRIX = None
FEATURE_DATA_CACHE = None


# -----------------------------
# Visualization helper
# -----------------------------
def plot_similar_images(input_images, top_similarities,
                        target_size=(512, 512)):
    """Visualize the query image(s) alongside the top-k similar results."""
    combined_width = target_size[0] * len(input_images)
    combined_image = Image.new("RGB", (combined_width, target_size[1]))
    for i, image in enumerate(input_images):
        img = Image.open(image).resize(target_size).convert("RGB")
        combined_image.paste(img, (i * target_size[0], 0))

    fig, axs = plt.subplots(
        1, len(top_similarities) + 2,
        figsize=(15, 5),
        gridspec_kw={"width_ratios": [1.5, 0.01] + [1] * len(top_similarities)},
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


# -----------------------------
# Data loading
# -----------------------------
def load_feature_data():
    """Load feature rows from disk once and cache them in memory."""
    global FEATURE_DATA_CACHE
    if FEATURE_DATA_CACHE is not None:
        return FEATURE_DATA_CACHE
    with open(FEATURE_DATA_PKL, "rb") as f:
        FEATURE_DATA_CACHE = pickle.load(f)
    return FEATURE_DATA_CACHE


# -----------------------------
# Histogram metric
# -----------------------------
def _bhattacharyya_distance(h1, h2, eps: float = 1e-12) -> float:
    """Bhattacharyya distance between two histograms (smaller = more similar)."""
    h1 = h1.astype(float)
    h2 = h2.astype(float)
    s1, s2 = h1.sum(), h2.sum()
    if s1 > 0:
        h1 = h1 / s1
    if s2 > 0:
        h2 = h2 / s2
    bc = np.sum(np.sqrt(np.clip(h1, 0, None) * np.clip(h2, 0, None)))
    return float(-np.log(bc + eps))


# -----------------------------
# FAISS index (IVF-Flat)
# -----------------------------
def _ensure_faiss_index(data, nlist=1000, nprobe=10):
    """
    Build an approximate FAISS IVF-Flat index over embeddings.
    nlist = number of clusters, nprobe = clusters searched at query.
    """
    global FAISS_INDEX, FAISS_IDS, FAISS_MATRIX, faiss
    if faiss is None:
        return False
    if FAISS_INDEX is not None:
        return True

    # Stack embeddings
    X = np.vstack([np.asarray(entry["embeddings"], dtype=np.float32)
                  for entry in data])
    FAISS_IDS = np.array([entry["image_id"] for entry in data], dtype=np.int64)

    # Normalize for cosine
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    FAISS_MATRIX = X / norms

    d = FAISS_MATRIX.shape[1]
    quantizer = faiss.IndexFlatIP(d)
    FAISS_INDEX = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)

    # Train IVF index
    FAISS_INDEX.train(FAISS_MATRIX)
    FAISS_INDEX.add(FAISS_MATRIX)

    # Set how many clusters to search
    FAISS_INDEX.nprobe = nprobe
    return True


# -----------------------------
# Main similarity function
# -----------------------------
def calculate_similarities(input_images, cursor, mode,
                           metric, top_k=5, verbose=False):
    """
    Compute top-k similar images under a chosen feature mode and metric.
    mode ∈ {'rgb','embeddings','ahashes','dhashes','phashes'}
    metric ∈ {'cosine','euclidean','manhattan','hamming','bhattacharyya','ann_cosine'}
    """
    global FEATURE_DATA_CACHE
    mode = mode.lower()
    metric = metric.lower()
    start_time = time.time()

    # Load features
    data = load_feature_data()

    # ---- Feature extraction for the query ----
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
        raise ValueError("Invalid mode.")

    # For hash modes, re-binarize
    if mode in {"ahashes", "dhashes", "phashes"}:
        input_transformed = (input_transformed > 0.5).astype(np.uint8)
        X = (X > 0.5).astype(np.uint8)

    # ---- Similarity computation ----
    if metric == "cosine":
        sims = cosine_similarity(input_transformed.reshape(1, -1), X).flatten()
        top_k_indices = np.argpartition(-sims, top_k)[:top_k]
        top_k_indices = top_k_indices[np.argsort(-sims[top_k_indices])]

    elif metric == "euclidean":
        sims = np.linalg.norm(X - input_transformed, axis=1)
        top_k_indices = np.argpartition(sims, top_k)[:top_k]
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])]

    elif metric == "manhattan":
        sims = np.sum(np.abs(X - input_transformed), axis=1)
        top_k_indices = np.argpartition(sims, top_k)[:top_k]
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])]

    elif metric == "hamming":
        sims = np.sum(input_transformed != X, axis=1)
        top_k_indices = np.argpartition(sims, top_k)[:top_k]
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])]

    elif metric == "bhattacharyya":
        if mode != "rgb":
            raise ValueError("Bhattacharyya only valid for mode='rgb'.")
        sims = np.array([_bhattacharyya_distance(input_transformed, row) for row in X])
        top_k_indices = np.argpartition(sims, top_k)[:top_k]
        top_k_indices = top_k_indices[np.argsort(sims[top_k_indices])]

    elif metric == "ann_cosine":
        if mode != "embeddings":
            raise ValueError("ann_cosine only valid with embeddings.")
        has_faiss = _ensure_faiss_index(data)
        q = np.asarray(input_transformed, dtype=np.float32)
        q /= (np.linalg.norm(q) + 1e-12)
        if has_faiss:
            D, I = FAISS_INDEX.search(q[None, :], top_k)
            sims = D[0]
            top_k_indices = I[0]
        else:
            # Fallback: NumPy cosine
            Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
            sims = (Xn @ q).astype(np.float32)
            top_k_indices = np.argpartition(-sims, top_k)[:top_k]
            top_k_indices = top_k_indices[np.argsort(-sims[top_k_indices])]

    else:
        raise ValueError("Invalid metric!")

    # ---- Retrieve paths ----
    top_similarities = []
    if len(sims) == len(data):
        for i in top_k_indices:
            image_id = data[i]["image_id"]
            sim_value = sims[i]
            file_path = fetch_image_path(image_id, cursor)
            if file_path:
                top_similarities.append((image_id, sim_value, file_path))
    else:
        for rank, i in enumerate(top_k_indices):
            image_id = data[i]["image_id"]
            sim_value = sims[rank]
            file_path = fetch_image_path(image_id, cursor)
            if file_path:
                top_similarities.append((image_id, sim_value, file_path))

    # ---- Plot results ----
    plot_similar_images(input_images, top_similarities)
    return top_similarities