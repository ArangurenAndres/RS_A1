import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os
import matplotlib.pyplot as plt

# Load data
def load_data(path="data_raw/ratings.dat"):
    df = pd.read_csv(path, sep="::", engine='python', names=["userId", "movieId", "rating", "timestamp"])
    return df

# Label positive interactions (rating >= 4)
def label_data(df):
    df["label"] = (df["rating"] >= 4).astype(int)
    return df

# Split positive + negative data before sampling
def split_data(df):
    # Split into train (80%) and temp (20%)
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
    # Split temp into validation (10%) and test (10%)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)
    return train_df, val_df, test_df

# Negative sampling for training set
def negative_sampling(train_df, all_items, num_neg_per_pos=1):
    user_pos_items = defaultdict(set)
    for row in train_df.itertuples():
        if row.label == 1:
            user_pos_items[row.userId].add(row.movieId)

    negative_samples = []
    for user in user_pos_items:
        pos_items = user_pos_items[user]
        for _ in range(len(pos_items) * num_neg_per_pos):
            neg_item = random.choice(list(all_items))
            while neg_item in pos_items:
                neg_item = random.choice(list(all_items))
            negative_samples.append((user, neg_item, 0))

    neg_df = pd.DataFrame(negative_samples, columns=["userId", "movieId", "label"])
    train_pos_df = train_df[train_df["label"] == 1][["userId", "movieId", "label"]]
    train_sampled_df = pd.concat([train_pos_df, neg_df], ignore_index=True)
    return train_sampled_df

def preprocess():
    df = load_data()
    df = label_data(df)

    # Keep only relevant columns
    df = df[["userId", "movieId", "label"]]

    # Full set of unique movies
    all_items = set(df["movieId"].unique())

    # Split data
    train_df, val_df, test_df = split_data(df)

    # Negative sampling only on train set
    train_final = negative_sampling(train_df, all_items, num_neg_per_pos=1)

    # For val/test we only use original labeled interactions (no sampling)
    val_final = val_df[["userId", "movieId", "label"]]
    test_final = test_df[["userId", "movieId", "label"]]

    return train_final, val_final, test_final

def plot_label_distribution(train, val, test):
    labels = ['Train', 'Validation', 'Test']
    pos_counts = [
        (train['label'] == 1).sum(),
        (val['label'] == 1).sum(),
        (test['label'] == 1).sum()
    ]
    neg_counts = [
        (train['label'] == 0).sum(),
        (val['label'] == 0).sum(),
        (test['label'] == 0).sum()
    ]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    ax.bar([i - width/2 for i in x], pos_counts, width, label='Positive (1)')
    ax.bar([i + width/2 for i in x], neg_counts, width, label='Negative (0)')

    ax.set_ylabel('Count')
    ax.set_title('Label Distribution in Train, Validation, and Test Sets')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train, val, test = preprocess()
    plot_label_distribution(train, val, test)


    #os.makedirs("data_processed", exist_ok=True)
    #train.to_csv("data_processed/train.csv", index=False)
    #val.to_csv("data_processed/val.csv", index=False)
    #test.to_csv("data_processed/test.csv", index=False)
    print("Preprocessing done. Files saved to 'data_processed/'")


