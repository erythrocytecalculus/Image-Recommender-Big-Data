from database_utils import create_images_table, insert_images, load_metadata, count_images, fetch_image_path
import sqlite3
import pytest
import os
import pickle
import sys

# database.py isn't in the same folder (kept separate for a cleaner structure)
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src")))


# Provide a test database connection
@pytest.fixture
def db_connection():
    """
    Creates a temporary in-memory SQLite database for tests.
    """
    # In-memory DB for temporary tests (no file on disk)
    conn = sqlite3.connect(":memory:")

    # Create the table
    create_images_table(conn)
    return conn


@pytest.fixture
def sample_image_data():
    """
    Supplies sample image metadata for tests.
    """
    return [
        {"image_id": 1, "root": "/test/path", "file": "image1.jpg"},
        {"image_id": 2, "root": "/test/path", "file": "image2.jpg"},
    ]


def test_create_table(db_connection):
    """
    Verifies that the table is created successfully.
    """
    conn = db_connection
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='images'")
    assert c.fetchone() is not None, "Table 'images' was not created"


def test_insert_and_count(db_connection, sample_image_data):
    """
    Tests inserting data and counting rows.
    """
    conn = db_connection
    insert_images(conn, sample_image_data)

    # Expect exactly two inserted rows
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM images")
    count = c.fetchone()[0]
    assert count == len(sample_image_data), f"Expected {
        len(sample_image_data)} rows, found: {count}"


def test_select_image_from_database(db_connection, sample_image_data):
    """
    Tests fetching image paths from the database.
    """
    conn = db_connection
    insert_images(conn, sample_image_data)
    c = conn.cursor()

    file_path = fetch_image_path(1, c)
    expected_path = os.path.normpath("/test/path/image1.jpg")
    file_path = os.path.normpath(file_path)

    assert file_path == expected_path, "Returned file path is incorrect"
