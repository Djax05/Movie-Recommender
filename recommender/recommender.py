import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional
from api.core.logging import get_logger


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

SENTENCE_EMBEDDING = DATA_DIR / "processed" / "sentence_embeddings.npy"
NUMERIC_SCALED = DATA_DIR / "processed" / "numeric_scaled.npy"
ENCODED_DATA = DATA_DIR / "processed" / "encoded_data.csv"
INDEX_FILE = DATA_DIR / "index" / "pynndescent_index.pkl"
TITLE_INDEX_FILE = DATA_DIR / "index" / "title_to_index.json"

SENTENCE_WEIGHT = 0.5
GENRE_WEIGHT = 0.3
NUMERIC_WEIGHT = 0.2

logger = get_logger(__name__)


class MovieRecommender:
    def __init__(self):
        logger.info("Loading Movie recommnder...")

        self.sentence_embeddings = np.load(SENTENCE_EMBEDDING).astype(
            "float32")
        self.encoded_data = pd.read_csv(ENCODED_DATA).values.astype(
            "float32")
        self.numeric_scaled = np.load(NUMERIC_SCALED).astype(
            "float32")

        self.w_sentence = SENTENCE_WEIGHT
        self.w_genre = GENRE_WEIGHT
        self.w_numeric = NUMERIC_WEIGHT

        with open(INDEX_FILE, 'rb') as f:
            self.index = pickle.load(f)

        with open(TITLE_INDEX_FILE, "r", encoding="utf-8") as f:
            title_to_index = json.load(f)

        self.title_to_index = {
            k.lower().strip(): v for k, v in title_to_index.items()
            }
        self.index_to_title = {v: k for k, v in self.title_to_index.items()}

        print(f"Loaded {len(self.title_to_index)} movies")
        logger.info("Movie recommender loaded")

    def get_recommendations(self,
                            title: str, k: int = 10
                            ) -> Optional[List[Dict]]:

        title = title.lower().strip()

        if title not in self.title_to_index:
            return None

        movie_id = self.title_to_index[title]

        query_sentence = self.sentence_embeddings[movie_id]
        query_genre = self.encoded_data[movie_id]
        query_numeric = self.numeric_scaled[movie_id]

        query_vec = np.hstack([
            query_sentence * self.w_sentence,
            query_genre * self.w_genre,
            query_numeric * self.w_numeric
        ]).astype("float32")

        query_vec = query_vec.reshape(1, -1)
        query_vec = normalize(query_vec)

        indices, distances = self.index.query(query_vec, k=k + 1)

        indices = indices[0][1:]
        distances = distances[0][1:]

        results = [
            {
                "title": self.index_to_title[int(idx)],
                "distance": float(dist)
            }
            for idx, dist in zip(indices, distances)
        ]

        return results

    def search_movies(self, query: str, limit: int = 10) -> List[str]:
        query = query.lower().strip()
        matches = []

        for title in self.title_to_index:
            if query in title.lower():
                matches.append(title)
                if len(matches) >= limit:
                    break

        return matches

    def movie_exists(self, title: str) -> bool:
        title_normalized = title.lower().strip()
        return title_normalized in self.title_to_index

    def get_total_movies(self) -> int:
        return int(len(self.title_to_index))


if __name__ == "__main__":
    print("Hello from movie_recommender!")

    recommender = MovieRecommender()

    results = recommender.get_recommendations("John Carter")

    if results:
        print("\n Recommendations for 'John Carter':")
        for i, rec in enumerate(results, 1):
            print(f" {i}. {rec['title']} (distance: {rec['distance']:.4f})")

    else:
        print('Movie not found')
