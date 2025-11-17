import numpy as np
import faiss
import json

embeddings = np.load("data/sentence_embeddings.npy").astype("float32")



def recommend_by_faiss(title, embeddings):

    title = title.lower().strip()
    with open("data/title_to_index.json", "r", encoding="utf-8") as f:
        title_to_index = json.load(f)

    title_to_index = {k.lower().strip(): v for k, v in title_to_index.items()}
    index_to_title = {int(v): k for k, v in title_to_index.items()}

    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    query_title = title
    query_id = title_to_index[query_title]

    query_vec = embeddings[query_id].reshape(1, -1)
    faiss.normalize_L2(query_vec)

    distances, indices = index.search(query_vec, 10)

    print("\nTop 10 recommendations for:", query_title)
    for i, (idx, score) in enumerate(zip(indices[0], distances[0])):
        print(f"{i+1}. {index_to_title[idx]} (score: {score})")


def main():
    print("Hello from movie-recommender!")
    # title = str(input("Enter a movie title: "))
    print(recommend_by_faiss("John Carter", embeddings))


if __name__ == "__main__":
    main()