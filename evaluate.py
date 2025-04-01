import numpy as np
import torch
import os
import json
from utils import preprocess as process
from utils import dataloader
from model.model import NCF
from sklearn.model_selection import train_test_split

def recall_at_k(recommended, ground_truth, k=10):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(ground_truth))
    return hits / len(ground_truth) if ground_truth else 0

def ndcg_at_k(recommended, ground_truth, k=10):
    recommended_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(recommended_k):
        if item in ground_truth:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))
    return dcg / idcg if idcg > 0 else 0

def evaluate(model, test_data, all_items, device, k=10):
    model.eval()
    user_group = test_data.groupby('userId')['movieId'].apply(set).to_dict()
    recall_scores = []
    ndcg_scores = []

    for user in user_group:
        ground_truth = user_group[user]
        item_tensor = torch.tensor(list(all_items), dtype=torch.long, device=device)
        user_tensor = torch.tensor([user] * len(all_items), dtype=torch.long, device=device)

        with torch.no_grad():
            scores = model(user_tensor, item_tensor)
            top_k_items = item_tensor[scores.argsort(descending=True)[:k]].tolist()

        recall = recall_at_k(top_k_items, ground_truth, k)
        ndcg = ndcg_at_k(top_k_items, ground_truth, k)

        recall_scores.append(recall)
        ndcg_scores.append(ndcg)

    avg_recall = np.mean(recall_scores)
    avg_ndcg = np.mean(ndcg_scores)
    return avg_recall, avg_ndcg

if __name__ == "__main__":
    # Paths
    data_path = "data_raw/ratings.dat"
    output_path = "data_preprocessed"
    config_path = "config.json"
    model_path = os.path.join("trained_models", "model.pth")

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    batch_size = config['batch_size']

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess and split data
    full_data = process.preprocess(data_path, output_path)
    train_data, temp_data = train_test_split(full_data, test_size=0.3, random_state=42, shuffle=True)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, shuffle=True)

    # Get number of users/items
    _, num_users, num_items = dataloader.get_dataloader_from_df(train_data, batch_size=batch_size, shuffle=True)

    # Load model
    model = NCF(num_users=num_users, num_items=num_items).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    # Evaluate
    all_items = full_data['movieId'].unique()
    avg_recall, avg_ndcg = evaluate(model, test_data, all_items, device, k=10)

    print(f"\nEvaluation Results:")
    print(f"Recall@10: {avg_recall:.4f}")
    print(f"NDCG@10:  {avg_ndcg:.4f}")
