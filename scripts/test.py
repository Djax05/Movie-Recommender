import numpy as np
import faiss

embeddings = np.load("data/sentence_embeddings.npy")
index = faiss.read_index("data/faiss_index.bin")

print("Embeddings shape:", embeddings.shape)
print("FAISS index dimension:", index.d)
