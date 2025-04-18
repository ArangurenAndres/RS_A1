from utils import dataloader as dataloader
from model.model import NCF
from train import train
from utils.data_preprocessing import load_and_preprocess_data  # <-- Change to actual file name
import torch
import json
import os

def run(data_path: str = None, output_path: str = None, config_path: str = None,
        model_name: str = "model.pth", results_path: str = "results", save_path: str = None,
        exp_name: str = None, ablation: bool = False):

    # 0. Load config file
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Training with the following parameters: {config}")

    # 1. Extract config parameters
    lr = config['lr']
    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    layers = config['layers']
    embedding_dim = config['embedding_dim']
    dropout_rate = config['dropout_rate']
    weight_decay = config['weight_decay']   

    # 2. Load and preprocess data
    train_df, val_df, test_df, num_users, num_items = load_and_preprocess_data(data_path)

    # 3. Create data loaders
    train_loader = dataloader.get_dataloader_from_df(train_df, batch_size=batch_size, shuffle=True)[0]
    val_loader = dataloader.get_dataloader_from_df(val_df, batch_size=batch_size, shuffle=False)[0]

    # 4. Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=embedding_dim,
        mlp_layers=layers,
        dropout_rate=dropout_rate
    ).to(device)

    # 5. Train the model
    os.makedirs(save_path, exist_ok=True)
    train_losses, val_losses = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
        num_epochs=num_epochs,
        weight_decay=weight_decay,
        save_path=save_path,
        model_name=model_name
    )

    # 6. Save training & validation losses
    os.makedirs(results_path, exist_ok=True)
    results = {
        "train_loss": train_losses,
        "val_loss": val_losses
    }

    if not ablation:
        with open(os.path.join(results_path, exp_name + ".json"), "w") as f:
            json.dump(results, f, indent=4)

    # 7. Save metadata (num_users, num_items) for evaluation
    metadata = {
        "num_users": int(num_users),
        "num_items": int(num_items)
    }
    meta_name = "meta_" + exp_name + ".json"
    with open(os.path.join(save_path, meta_name), "w") as f:
        json.dump(metadata, f, indent=4)

    return train_losses, val_losses


if __name__ == "__main__":
    data_path = "data_raw/ratings.dat"
    output_path = "data_preprocessed"  # Not used in current version, but kept for consistency
    config_path = "config.json"
    results_path = "results"
    save_path = "trained_models"

    # specify the experiment name here
    exp_name = "exp_test"

    # specify the model name here 
    model_name = "model_test.pth"

    run(
        data_path=data_path,
        output_path=output_path,
        config_path=config_path,
        results_path=results_path,
        save_path=save_path,
        exp_name=exp_name,
        ablation=False,
        model_name=model_name
    )
