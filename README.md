# image_recommender

image_recommender/
├── data/                   # image files or links to paths
├── db/                    # SQLite or PostgreSQL setup
│   └── image_meta.db
├── images/                # (optional) image cache
├── embeddings/            # Precomputed deep features
├── src/
│   ├── database.py        # handle ID ↔ metadata ↔ path
│   ├── generator.py       # image loading pipeline (PIL/OpenCV)
│   ├── similarity/
│   │   ├── color.py       # color-based metric
│   │   ├── embedding.py   # DL model-based similarity
│   │   ├── custom.py      # your choice (e.g. hashing)
│   ├── search.py          # ANN / brute-force search
│   ├── recommender.py     # core logic: combine inputs, score, rank
│   └── utils.py           # timing, logging, etc.
├── tests/
│   ├── test_similarity.py
│   └── test_database.py
├── requirements.txt
└── main.py                # command line or GUI runner
