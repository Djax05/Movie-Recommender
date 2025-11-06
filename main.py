import pandas as pd


data_sample = pd.read_csv("data/data_sample.csv")


def recommend_within_cluster(title):
    title = title.strip().lower()
    data_sample['movie_title_clean'] = data_sample['movie_title'].str.strip().str.lower()

    matches = data_sample[data_sample['movie_title_clean'] == title]
    if matches.empty:
        return f"No movie found for '{title}'. Try a different title."

    cluster = matches['cluster_id'].values[0]
    similar_movies = data_sample[data_sample['cluster_id'] == cluster]['movie_title'].tolist()
    similar_movies.remove(matches['movie_title'].values[0])
    return similar_movies


def main():
    print("Hello from movie-recommender!")
    print(recommend_within_cluster("John Carter"))


if __name__ == "__main__":
    main()
