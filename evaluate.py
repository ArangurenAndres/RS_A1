import torch
import pandas as pd
import numpy as np
import json
import random
from tqdm import tqdm
import os
from model.model import NCF
from utils import preprocess_test as process  # or kerem_preprocess

def recall_at_k(ranked_list, ground_truth, k):
    return int(ground_truth in ranked_list[:k])

def ndcg_at_k(ranked_list, ground_truth, k):
    if ground_truth in ranked_list[:k]:
        index = ranked_list.index(ground_truth)
        return 1 / np.log2(index + 2)
    return 0.0

def evaluate(model, test_df, all_items, device, k=10, num_negative=99):
    model.eval()
    user_item_dict = test_df.groupby("userId")["movieId"].apply(set).to_dict()

    recalls = []
    ndcgs = []

    users = list(user_item_dict.keys())

    with torch.no_grad():
        for user in tqdm(users, desc="Evaluating"):
            pos_items = list(user_item_dict[user])
            if not pos_items:
                continue
            pos_item = random.choice(pos_items)

            neg_items = set(all_items) - set(pos_items)
            if len(neg_items) < num_negative:
                continue
            sampled_neg_items = random.sample(list(neg_items), num_negative)

            test_items = sampled_neg_items + [pos_item]
            item_tensor = torch.tensor(test_items, dtype=torch.long).to(device)
            user_tensor = torch.tensor([user] * len(test_items), dtype=torch.long).to(device)

            scores = model(user_tensor, item_tensor).cpu().numpy()
            ranked_items = [x for _, x in sorted(zip(scores, test_items), reverse=True)]

            recalls.append(recall_at_k(ranked_items, pos_item, k))
            ndcgs.append(ndcg_at_k(ranked_items, pos_item, k))

    avg_recall = np.mean(recalls)
    avg_ndcg = np.mean(ndcgs)
    return avg_recall, avg_ndcg

def run_evaluation(model_path="trained_models/model_test.pth",
                   config_path="config.json",
                   results_path="results/eval_metrics.json",
                   meta_path="trained_models/meta.json",
                   k=10,
                   num_negative=99):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model metadata (num_users, num_items)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    num_users = meta["num_users"]
    num_items = meta["num_items"]

    # Load model config (to get embedding/layer info)
    with open(config_path, "r") as f:
        config = json.load(f)

    # Preprocess test data
    train_df, val_df, test_df = process.preprocess()
    all_items = pd.concat([train_df, val_df, test_df])["movieId"].unique()

    # Reconstruct model
    model = NCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=config["embedding_dim"],
        mlp_layers=config["layers"],
        dropout_rate=config["dropout_rate"]
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    # Run evaluation
    recall, ndcg = evaluate(model, test_df, all_items, device, k=k, num_negative=num_negative)
    print(f"Recall@{k}: {recall:.4f}")
    print(f"NDCG@{k}: {ndcg:.4f}")

    # Save evaluation results
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"Recall@10": recall, "NDCG@10": ndcg}, f, indent=4)

    return recall, ndcg

if __name__ == "__main__":
    model_path = "trained_models/model_test.pth"
    config_path = "config.json"
    results_path = "results/eval_metrics.json"
    meta_path = "trained_models/meta.json"
    run_evaluation()
