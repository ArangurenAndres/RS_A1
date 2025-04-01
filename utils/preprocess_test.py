import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import time

def preprocess(data_path, output_path, save=True, negative_ratio=1):
    """
    Preprocess the MovieLens 1M dataset and split it into train, validation, and test sets.
    
    Args:
        data_path: Path to the ratings.dat file
        output_path: Directory to save processed data
        save: Whether to save the processed data
        negative_ratio: Number of negative samples per positive sample
        
    Returns:
        train_df, val_df, test_df: Split DataFrames
    """
    print("Loading and preprocessing MovieLens 1M dataset...")
    
    # Load ratings data
    ratings_cols = ['userId', 'movieId', 'rating', 'timestamp']
    ratings_df = pd.read_csv(
        data_path, 
        sep='::', 
        header=None, 
        names=ratings_cols,
        engine='python'  # Use python engine for custom separator
    )
    
    print(f"Loaded {len(ratings_df)} ratings from {len(ratings_df['userId'].unique())} users on {len(ratings_df['movieId'].unique())} movies.")
    
    # Convert explicit ratings to implicit feedback (rating >= 4 is positive)
    ratings_df['label'] = (ratings_df['rating'] >= 4).astype(int)
    
    # Keep only positive interactions for now
    positive_df = ratings_df[ratings_df['label'] == 1][['userId', 'movieId', 'label', 'timestamp']]
    print(f"Found {len(positive_df)} positive interactions.")
    
    # Generate negative samples
    negative_df = generate_negative_samples(ratings_df, negative_ratio)
    print(f"Generated {len(negative_df)} negative samples.")
    
    # Combine positive and negative interactions
    processed_df = pd.concat([positive_df, negative_df], ignore_index=True)
    
    # Split the data
    print("\nSplitting data into train, validation, and test sets...")
    train_df, val_df, test_df = split_data(processed_df)
    
    # Check the distribution of labels in each split
    print("\nPositive interactions ratio:")
    print(f"Train: {train_df['label'].mean():.2f}")
    print(f"Validation: {val_df['label'].mean():.2f}")
    print(f"Test: {test_df['label'].mean():.2f}")
    
    # Save the processed data if requested
    if save:
        save_splits(train_df, val_df, test_df, output_path)
    
    return train_df, val_df, test_df

def generate_negative_samples(ratings_df, negative_ratio=1):
    """
    Generate negative samples through random sampling - IMPROVED VERSION.
    
    Args:
        ratings_df: DataFrame containing all ratings
        negative_ratio: Number of negative samples per positive sample
        
    Returns:
        negative_df: DataFrame containing negative samples
    """
    # Create a set of all user-movie pairs for fast lookup
    print("Creating user-movie interaction set...")
    user_movie_set = set(zip(ratings_df['userId'], ratings_df['movieId']))
    
    # Get all unique users and movies
    all_users = ratings_df['userId'].unique()
    all_movies = ratings_df['movieId'].unique()
    
    # Count positive interactions to determine number of negatives to generate
    positive_count = sum(ratings_df['rating'] >= 4)
    num_negative_samples = positive_count * negative_ratio
    print(f"Aiming to generate {num_negative_samples} negative samples...")
    
    # Initialize negative samples list
    negative_samples = []
    
    # Track progress
    start_time = time.time()
    last_update = start_time
    
    # Random sampling approach
    np.random.seed(42)  # For reproducibility
    
    # Pre-generate a large batch of random pairs and filter them
    print("Generating negative samples...")
    attempts = 0
    max_attempts = num_negative_samples * 5  # Limit total attempts to avoid infinite loops
    
    while len(negative_samples) < num_negative_samples and attempts < max_attempts:
        # Generate a large batch of random pairs at once (vectorized operation)
        batch_size = min(100000, num_negative_samples - len(negative_samples))
        
        random_users = np.random.choice(all_users, batch_size)
        random_movies = np.random.choice(all_movies, batch_size)
        
        # Process batch
        for i in range(batch_size):
            attempts += 1
            user_id = random_users[i]
            movie_id = random_movies[i]
            
            # Check if this is a valid negative sample
            if (user_id, movie_id) not in user_movie_set:
                negative_samples.append({
                    'userId': user_id,
                    'movieId': movie_id,
                    'label': 0,
                    'timestamp': ratings_df['timestamp'].mean()
                })
                user_movie_set.add((user_id, movie_id))  # Prevent future duplicates
            
            # Print progress every 5 seconds
            current_time = time.time()
            if current_time - last_update > 5:
                print(f"Progress: {len(negative_samples)}/{num_negative_samples} negative samples generated ({len(negative_samples)/num_negative_samples:.1%})")
                last_update = current_time
                
            # Break if we have enough samples
            if len(negative_samples) >= num_negative_samples:
                break
    
    if len(negative_samples) < num_negative_samples:
        print(f"Warning: Only generated {len(negative_samples)} of {num_negative_samples} requested negative samples after {attempts} attempts.")
    
    print(f"Negative sampling completed in {time.time() - start_time:.1f} seconds")
    
    # Convert to DataFrame
    negative_df = pd.DataFrame(negative_samples)
    return negative_df

def split_data(df, stratify_col=None):  # Changed default to None
    """
    Split the dataset into training (70%), validation (15%), and testing (15%) sets.
    
    Args:
        df: DataFrame with processed data
        stratify_col: Column to use for stratified sampling (if None, no stratification)
        
    Returns:
        train_df, val_df, test_df: Split DataFrames
    """
    # First split: 70% train, 30% temp
    stratify = df[stratify_col] if stratify_col else None
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.3,
        random_state=42,
        stratify=stratify
    )
    
    # Second split: 15% validation, 15% test (half of the 30% temp)
    stratify = temp_df[stratify_col] if stratify_col else None
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.5,
        random_state=42,
        stratify=stratify
    )
    
    print(f"Data split: Train: {len(train_df)} ({len(train_df)/len(df):.1%}), "
          f"Validation: {len(val_df)} ({len(val_df)/len(df):.1%}), "
          f"Test: {len(test_df)} ({len(test_df)/len(df):.1%})")
    
    return train_df, val_df, test_df

def save_splits(train_df, val_df, test_df, output_dir):
    """Save the data splits to CSV files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(output_dir, 'validation.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    
    print(f"Saved processed data to {output_dir}")

if __name__ == "__main__":
    # For standalone running of this script
    data_path = "data_raw/ratings.dat"  # Path to your ratings.dat file
    output_path = "data_preprocessed"  # Directory to save processed data
    
    # You can adjust these parameters
    negative_ratio = 1  # Number of negative samples per positive sample
    
    print(f"Starting preprocessing with negative_ratio={negative_ratio}")
    try:
        train_df, val_df, test_df = preprocess(
            data_path=data_path,
            output_path=output_path,
            save=True,
            negative_ratio=negative_ratio
        )
        print("Preprocessing completed successfully!")
        print(f"Generated datasets: Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")