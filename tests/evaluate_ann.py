import numpy as np
import pickle
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

sentence_embeddings = np.load("data/processed/sentence_embeddings.npy")
encoded_data = pd.read_csv("data/processed/encoded_data.csv").values.astype(
    "float32")
numeric_scaled = np.load("data/processed/numeric_scaled.npy").astype("float32")

w_sentence = 0.5
w_genre = 0.3
w_numeric = 0.2

sentence_w = sentence_embeddings * w_sentence
genre_w = encoded_data * w_genre
numeric_w = numeric_scaled * w_numeric

full_features = np.hstack([
    sentence_w,
    genre_w,
    numeric_w
])

full_features_norm = normalize(full_features)

with open("data/index/pynndescent_index.pkl", 'rb') as f:
    index = pickle.load(f)

with open("data/index/title_to_index.json", "r", encoding="utf-8") as f:
    title_to_index = json.load(f)

title_to_index = {k.lower().strip(): v for k, v in title_to_index.items()}
index_to_title = {v: k for k, v in title_to_index.items()}


def recall_at_k(movie_title, k=10):
    movie_title = movie_title.lower().strip()

    if movie_title not in title_to_index:
        return f"{movie_title} not found"

    movie_id = title_to_index[movie_title]
    query_vec = full_features_norm[movie_id].reshape(1, -1)

    sims = cosine_similarity(query_vec, full_features_norm)[0]
    true_top_k = sims.argsort()[::-1][1:k+1]

    ann_indices, _ = index.query(query_vec, k=k+1)
    ann_top_k = ann_indices[0][1:]
    print(ann_top_k)

    true_set = set(true_top_k)
    ann_set = set(ann_top_k)

    recall = len(true_set & ann_set) / len(true_set)

    titles = [index_to_title[i] for i in ann_top_k]

    return recall, titles


if __name__ == "__main__":
    movie = "John Carter"
    recall, titles = recall_at_k(movie, k=10)

    print(f"\nRecall@10 for '{movie}': {recall:.3f}")
    print("\nANN neighbors:")

    for t in titles:
        print(" -", t)
