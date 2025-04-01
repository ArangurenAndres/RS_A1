import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class InteractionDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['userId'].values, dtype=torch.long)
        self.items = torch.tensor(df['movieId'].values, dtype=torch.long)
        self.labels = torch.tensor(df['label'].values, dtype=torch.float)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

def get_dataloader_from_df(df, batch_size=256, shuffle=False):
    dataset = InteractionDataset(df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    num_users = df['userId'].max() + 1
    num_items = df['movieId'].max() + 1

    return dataloader, num_users, num_items


if __name__ == "__main__":
    # Example: preprocess and use in memory
    from preprocess import preprocess  # assumes preprocess.py is in the same dir

    full_df = preprocess(data_path="data_raw/ratings.dat", output_path="data_preprocessed", save=False)

    # Shuffle and split the dataframe manually if needed
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(full_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    train_loader, num_users, num_items = get_dataloader_from_df(train_df, batch_size=256, shuffle=True)
    for users, items, labels in train_loader:
        print(users.shape, items.shape, labels.shape)
        break
