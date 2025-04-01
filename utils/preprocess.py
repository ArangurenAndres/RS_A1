import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def preprocess(data_path: str=None, output_path: str=None, save=False, neg_ratio=4):
    # Load ratings
    ratings = pd.read_csv(data_path, sep='::', engine='python', 
                      names=['userId', 'movieId', 'rating', 'timestamp'])

    # Filter positive interactions (implicit feedback)
    positive_ratings = ratings[ratings['rating'] >= 4].copy()
    positive_ratings['label'] = 1

    # Step 1: Split only the positives first
    train_pos, temp_pos = train_test_split(positive_ratings, test_size=0.3, random_state=42, shuffle=True)
    val_pos, test_pos = train_test_split(temp_pos, test_size=0.5, random_state=42, shuffle=True)

    all_movie_ids = set(ratings['movieId'].unique())

    # Step 2: Generate negatives separately for each split
    def sample_negatives(pos_df):
        user_positive = pos_df.groupby('userId')['movieId'].apply(set).to_dict()
        negative_samples = []

        for user, pos_movies in user_positive.items():
            negatives = list(all_movie_ids - pos_movies)
            sample_size = min(len(pos_movies) * neg_ratio, len(negatives))
            sampled_negatives = np.random.choice(negatives, size=sample_size, replace=False)
            for movie in sampled_negatives:
                negative_samples.append([user, movie, 0])
        
        neg_df = pd.DataFrame(negative_samples, columns=['userId', 'movieId', 'label'])
        pos_df = pos_df[['userId', 'movieId', 'label']]
        return pd.concat([pos_df, neg_df], ignore_index=True)

    train_df = sample_negatives(train_pos)
    val_df = sample_negatives(val_pos)
    test_df = sample_negatives(test_pos)

    # Step 3: Save if required
    os.makedirs(output_path, exist_ok=True)
    if save:
        print("Saving processed data to CSV files...")
        train_df.to_csv(f"{output_path}/train.csv", index=False)
        val_df.to_csv(f"{output_path}/val.csv", index=False)
        test_df.to_csv(f"{output_path}/test.csv", index=False)

    return train_df, val_df, test_df


if __name__ == "__main__":
    data_path = "data_raw/ratings.dat"
    output_path = "data_preprocessed"
    train_df, val_df, test_df = preprocess(data_path, output_path, save=True)
