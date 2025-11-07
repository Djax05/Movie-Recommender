import pandas as pd
import numpy as np

data = pd.read_csv("data/data_sample.csv")
cosine_sim = np.load("data/cosine_sim.npy")


def recommend_by_cosine(title, data, cosine_sim, top_n=10):
    title = title.strip().lower()
    data['movie_title_clean'] = data['movie_title'].str.strip().str.lower()
    matches = data[data['movie_title_clean'] == title]

    if matches.empty:
        return f"No movie found for {title}"

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]

    results = [(data["movie_title"].iloc[i], score) for i, score in sim_scores]
    return results


def main():
    print("Hello from movie-recommender!")
    title = str(input("Enter a movie title: "))
    print(recommend_by_cosine(title, data, cosine_sim))


if __name__ == "__main__":
    main()
