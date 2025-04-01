from utils import dataloader as dataloader
from model.model import NCF
from train import train
from evaluate import evaluate
import torch
import json
import os
import pandas as pd

def run(output_path: str = None, config_path: str = None,
        results_path: str = "results", save_path: str = None):

    # 0. Load config file
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Training with following parameters: {config}")

    lr = config['lr']
    num_epochs = config['num_epochs']
    patience = config['patience']
    batch_size = config['batch_size']
    layers = config['layers']
    embedding_dim = config['embedding_dim']

    # 1. Load preprocessed CSV files
    train_df = pd.read_csv(os.path.join(output_path, "train.csv"))
    val_df = pd.read_csv(os.path.join(output_path, "validation.csv"))
    test_df = pd.read_csv(os.path.join(output_path, "test.csv"))

    # 2. Load data loaders
    train_loader, num_users, num_items = dataloader.get_dataloader_from_df(train_df, batch_size=batch_size, shuffle=True)
    val_loader, _, _ = dataloader.get_dataloader_from_df(val_df, batch_size=batch_size, shuffle=False)

    # 3. Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NCF(num_users=num_users, num_items=num_items,
                embedding_dim=embedding_dim, mlp_layers=layers).to(device)

    # 4. Train the model
    os.makedirs(save_path, exist_ok=True)
    train_losses, val_losses = train(model, train_loader, val_loader, device,
                                     lr=lr, num_epochs=num_epochs, patience=patience,
                                     use_early_stopping=False, save_path=save_path)

    # 5. Save training loss results
    os.makedirs(results_path, exist_ok=True)
    results = {
        "train_loss": train_losses,
        "val_loss": val_losses
    }
    with open(os.path.join(results_path, "loss_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # 6. (Optional) Evaluate on test set
    # model_path = os.path.join(save_path, "model.pth")
    # model.load_state_dict(torch.load(model_path, map_location=device))
    # model.eval()
    # all_items = pd.concat([train_df, val_df, test_df])['movieId'].unique()
    # avg_recall, avg_ndcg = evaluate(model, test_df, all_items, device, k=10)
    # print(f"Test Recall@10: {avg_recall:.4f}, NDCG@10: {avg_ndcg:.4f}")


if __name__ == "__main__":
    output_path = "data_preprocessed"
    config_path = "config.json"
    results_path = "results"
    save_path = "trained_models"
    run(output_path=output_path, config_path=config_path,
        results_path=results_path, save_path=save_path)
