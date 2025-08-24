import os
import sqlite3
import pickle
from config_paths import IMAGE_DATA_PKL, DB_PATH


def create_images_table(conn):
    """Create the images table if it doesn't already exist."""
    cursor = conn.cursor()
    cursor.execute(
        '''CREATE TABLE IF NOT EXISTS images
           (image_id INTEGER PRIMARY KEY, root TEXT, file TEXT)'''
    )
    conn.commit()


def insert_images(conn, records):
    """Insert multiple image records into the database."""
    cursor = conn.cursor()
    entries = [(d["image_id"], d["root"], d["file"]) for d in records]
    cursor.executemany('INSERT INTO images VALUES (?,?,?)', entries)
    conn.commit()


def load_metadata(metadata_file):
    """Load metadata from the pickle file."""
    with open(metadata_file, "rb") as f:
        return pickle.load(f)


def count_images(conn):
    """Return total number of rows in the images table."""
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM images')
    total = cursor.fetchone()[0]
    conn.close()
    return total


def fetch_image_path(image_id, cursor):
    """Fetch an image path by image_id."""
    cursor.execute('SELECT root, file FROM images WHERE image_id = ?', (image_id,))
    result = cursor.fetchone()
    if result:
        root, file = result
        return os.path.join(root, file)


def build_database(db_path, metadata_file):
    """Initialize database and insert image metadata."""
    conn = sqlite3.connect(db_path)
    create_images_table(conn)
    data = load_metadata(metadata_file)
    insert_images(conn, data)
    total_rows = count_images(conn)
    print(f"Inserted data successfully. Total rows: {total_rows}")
    conn.close()


if __name__ == "__main__":
    build_database(DB_PATH, IMAGE_DATA_PKL)
