import faiss
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize


sentence_embeddings = np.load("data/sentence_embeddings.npy").astype("float32")
encoded_data = pd.read_csv("data/encoded_data.csv").astype("float32")
numeric_scaled = np.load("data/numeric_scaled.npy").astype("float32")

print(sentence_embeddings.shape)
print(encoded_data.shape)
print(numeric_scaled.shape)


w_sentence = 0.5
w_genre = 0.3
w_numeric = 0.2


sentence_w = sentence_embeddings * w_sentence
genre_w = encoded_data.values * w_genre
numeric_w = numeric_scaled * w_numeric

full_features = np.hstack([
    sentence_w,
    genre_w,
    numeric_w
])

full_features_norm = normalize(full_features)

index = faiss.IndexFlatIP(full_features_norm.shape[1])

index.add(full_features_norm.astype("float32"))

faiss.write_index(index, "data/faiss_index.bin")

print("Faiss Index saved")
