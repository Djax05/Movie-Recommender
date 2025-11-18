import faiss
import numpy as np


X = np.load("data/sentence_embeddings.npy").astype("float32")

faiss.normalize_L2(X)

dim = X.shape[1]

index = faiss.IndexFlatIP(dim)

index.add(X)

faiss.write_index(index, "data/faiss_index.bin")

print("Faiss Index saved")