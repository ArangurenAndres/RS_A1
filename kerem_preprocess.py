import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess(data_path, output_path, save=False):
    """
    Preprocess the MovieLens dataset.
    
    Parameters:
    - data_path: Path to the ratings file (e.g., "data_raw/ratings.dat").
                 Assumes that movies.dat is in the same folder.
    - output_path: Directory to save the preprocessed data if save=True.
    - save: Boolean flag indicating whether to save the train/val/test splits as CSV files.
    
    Returns:
    - train: Training DataFrame
    - val: Validation DataFrame
    - test: Test DataFrame
    """
    # Determine the directory and construct the movies file path
    data_dir = os.path.dirname(data_path)
    movies_path = os.path.join(data_dir, 'movies.dat')
    
    # -------------------------------
    # 1. Load the Data
    # -------------------------------
    ratings = pd.read_csv(data_path, sep='::', engine='python',
                          names=['userId', 'movieId', 'rating', 'timestamp'], encoding='latin1')
    movies = pd.read_csv(movies_path, sep='::', engine='python',
                         names=['movieId', 'title', 'genres'], encoding='latin1')
    
    # -------------------------------
    # 2. Convert to Implicit Feedback
    # -------------------------------
    ratings['label'] = (ratings['rating'] >= 4).astype(int)
    positive_interactions = ratings[ratings['label'] == 1][['userId', 'movieId', 'label']]
    
    # -------------------------------
    # 3. Negative Sampling
    # -------------------------------
    # Get a set of all movieIds from the movies dataset
    all_movie_ids = set(movies['movieId'].unique())
    
    # Map each user to the set of movies they have positively interacted with
    user_positive = positive_interactions.groupby('userId')['movieId'].apply(set).to_dict()
    
    # Generate negative samples: for each user, sample as many negatives as positives
    negative_samples = []
    for user, pos_movies in user_positive.items():
        num_negatives = len(pos_movies)
        # Determine movies that the user has NOT interacted with
        neg_pool = list(all_movie_ids - pos_movies)
        if len(neg_pool) >= num_negatives:
            sampled_negatives = np.random.choice(neg_pool, size=num_negatives, replace=False)
        else:
            sampled_negatives = np.random.choice(neg_pool, size=num_negatives, replace=True)
        for movie in sampled_negatives:
            negative_samples.append({'userId': user, 'movieId': movie, 'label': 0})
    
    negative_interactions = pd.DataFrame(negative_samples)
    
    # -------------------------------
    # 4. Combine Positive and Negative Interactions
    # -------------------------------
    data = pd.concat([positive_interactions, negative_interactions], ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # -------------------------------
    # 5. Split into Train, Validation, and Test Sets
    # -------------------------------
    train_val, test = train_test_split(data, test_size=0.15, random_state=42)
    # Note: To have 70% training overall and 15% validation, the validation split is about 17.65% of train_val.
    train, val = train_test_split(train_val, test_size=0.1765, random_state=42)
    
    # -------------------------------
    # 6. Optionally Save the Data
    # -------------------------------
    if save:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        train.to_csv(os.path.join(output_path, 'train.csv'), index=False)
        val.to_csv(os.path.join(output_path, 'val.csv'), index=False)
        test.to_csv(os.path.join(output_path, 'test.csv'), index=False)
    
    return train, val, test

def plot_label_distribution(train, val, test):
    """
    Plot the distribution of labels (positive vs negative) across the train, validation, and test splits.
    """
    # Add a 'split' indicator for plotting
    train['split'] = 'train'
    val['split'] = 'val'
    test['split'] = 'test'
    
    combined = pd.concat([train, val, test], ignore_index=True)
    
    plt.figure(figsize=(8, 6))
    sns.countplot(data=combined, x='split', hue='label')
    plt.title('Label Distribution Across Splits')
    plt.xlabel('Data Split')
    plt.ylabel('Count')
    plt.legend(title='Label', labels=['Negative (0)', 'Positive (1)'])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_path = "data_raw/ratings.dat"
    output_path = "data_preprocessed"
    train_df, val_df, test_df = preprocess(data_path, output_path, save=False)
    plot_label_distribution(train_df, val_df, test_df)
