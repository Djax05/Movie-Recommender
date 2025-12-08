import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
from pynndescent import NNDescent

sentence_embeddings = np.load("data/processed/sentence_embeddings.npy").astype(
    "float32")
encoded_data = pd.read_csv("data/processed/encoded_data.csv").astype("float32")
numeric_scaled = np.load("data/processed/numeric_scaled.npy").astype("float32")

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

index = NNDescent(
    full_features_norm,
    metric="cosine",
    n_neighbors=15,
    n_jobs=-1
)

with open("data/index/pynndescent_index.pkl", 'wb') as f:
    pickle.dump(index, f)

print("Pynndescent Index saved")
