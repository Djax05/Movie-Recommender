"""
Configuration for the movie recommender API.

"""
from pathlib import Path
from pydantic_settings import BaseSettings


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Path
DATA_DIR = BASE_DIR / "data"

# Processed Data Paths
SENTENCE_EMBEDDINGS = DATA_DIR / "processed" / "sentence_embeddings.npy"
NUMERIC_SCALED = DATA_DIR / "processed" / "numeric_scaled.npy"
ENCODED_DATA = DATA_DIR / "processed" / "encoded_data.csv"

# Index Paths
INDEX_FILE = DATA_DIR / "index" / "pynndescent_index.pkl"
TITLE_INDEX_FILE = DATA_DIR / "index" / "title_to_index.json"

# Model hyperparameters
W_SENTENCE = 0.5
W_GENRE = 0.3
W_NUMERIC = 0.2


class Settings(BaseSettings):

    api_title: str = "Movie recommendation API"
    api_version: str = "1.0.0"
    api_description: str = "Content based movie recomendation using ANN"

    min_recommendations: int = 1
    max_recommendations: int = 50
    default_recommendations: int = 10

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    class Config:
        env_file = ".env"


settings = Settings()
