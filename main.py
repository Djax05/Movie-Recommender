import numpy as np
import pandas as pd
import faiss
import json

sentence_embeddings = np.load("data/sentence_embeddings.npy").astype("float32")
encoded_data = pd.read_csv("data/encoded_data.csv").values.astype("float32")
numeric_scaled = np.load("data/numeric_scaled.npy").astype("float32")

w_sentence = 0.8
w_genre = 0.5
w_numeric = 0.2

sentence_w = sentence_embeddings * w_sentence
genre_w = encoded_data * w_genre
numeric_w = numeric_scaled * w_numeric

full_features = np.hstack([
    sentence_w,
    genre_w,
    numeric_w
])

faiss.normalize_L2(full_features)

index = faiss.read_index("data/faiss_index.bin")

with open("data/title_to_index.json", "r", encoding="utf-8") as f:
    title_to_index = json.load(f)

title_to_index = {k.lower().strip(): v for k, v in title_to_index.items()}

index_to_title = {v: k for k, v in title_to_index.items()}


def recommend(title, k=10):
    title = title.lower().strip()

    if title not in title_to_index:
        return f"Movie '{title}' not found."

    movie_id = title_to_index[title]

    query_sentence = sentence_embeddings[movie_id]
    query_genre = encoded_data[movie_id]
    query_numeric = numeric_scaled[movie_id]
    query_vec = np.hstack([
        query_sentence * w_sentence,
        query_genre * w_genre,
        query_numeric * w_numeric
    ]).astype("float32")

    query_vec = query_vec.reshape(1, -1)

    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, k + 1)

    indices = indices[0][1:]
    distances = distances[0][1:]

    results = [(index_to_title[int(idx)], float(dist)) for idx, dist in zip(indices, distances)]
    return results


def main():
    print("Hello from movie-recommender!")
    # title = str(input("Enter a movie title: "))
    print(recommend("John Carter"))


if __name__ == "__main__":
    main()