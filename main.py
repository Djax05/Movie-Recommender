import numpy as np
import faiss
import json

embeddings = np.load("data/sentence_embeddings.npy").astype("float32")
faiss.normalize_L2(embeddings)

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

    query_vec = embeddings[movie_id].reshape(1, -1)
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