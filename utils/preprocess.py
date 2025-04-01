import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def preprocess(data_path: str=None, output_path: str=None, save=False, neg_ratio=1):
    # Load ratings
    ratings = pd.read_csv(data_path, sep='::', engine='python', 
                      names=['userId', 'movieId', 'rating', 'timestamp'])

    # Create binary labels (>=4 is positive)
    ratings['label'] = (ratings['rating'] >= 4).astype(int)
    
    # Get all user-movie pairs
    user_movie_pairs = ratings[['userId', 'movieId']].drop_duplicates()
    
    # Step 1: Split all data first to prevent leakage
    train, temp = train_test_split(user_movie_pairs, test_size=0.3, random_state=42, stratify=ratings['userId'])
    val, test = train_test_split(temp, test_size=0.5, random_state=42, stratify=temp['userId'])
    
    # Get positive interactions for each split
    def get_positives(split_df):
        return pd.merge(split_df, ratings, on=['userId', 'movieId'])[['userId', 'movieId', 'label']]
    
    train_pos = get_positives(train)
    val_pos = get_positives(val)
    test_pos = get_positives(test)
    
    # Get all movies each user has interacted with (positive or negative)
    user_interacted_movies = ratings.groupby('userId')['movieId'].apply(set).to_dict()
    all_movie_ids = set(ratings['movieId'].unique())
    
    # Function to sample negatives ensuring no data leakage
    def sample_negatives(pos_df, user_history):
        negative_samples = []
        
        for user in pos_df['userId'].unique():
            pos_movies = set(pos_df[pos_df['userId'] == user]['movieId'])
            user_negatives = list(all_movie_ids - user_history[user])
            
            # Sample negatives
            sample_size = min(len(pos_movies) * neg_ratio, len(user_negatives))
            sampled_negatives = np.random.choice(user_negatives, size=sample_size, replace=False)
            
            for movie in sampled_negatives:
                negative_samples.append([user, movie, 0])
        
        neg_df = pd.DataFrame(negative_samples, columns=['userId', 'movieId', 'label'])
        return pd.concat([pos_df, neg_df], ignore_index=True)
    
    # Generate negatives for each split
    train_df = sample_negatives(train_pos, user_interacted_movies)
    val_df = sample_negatives(val_pos, user_interacted_movies)
    test_df = sample_negatives(test_pos, user_interacted_movies)
    
    # Shuffle the data
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save if required
    if save:
        os.makedirs(output_path, exist_ok=True)
        print("Saving processed data to CSV files...")
        train_df.to_csv(f"{output_path}/train.csv", index=False)
        val_df.to_csv(f"{output_path}/val.csv", index=False)
        test_df.to_csv(f"{output_path}/test.csv", index=False)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    data_path = "data_raw/ratings.dat"
    output_path = "data_preprocessed"
    train_df, val_df, test_df = preprocess(data_path, output_path, save=True)