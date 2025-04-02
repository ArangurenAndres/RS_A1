import torch
import os
import torch.nn as nn
from tqdm import trange, tqdm
from utils import preprocess as process
from utils import dataloader as dataloader
from model.model import NCF
from sklearn.model_selection import train_test_split


def train(model, train_loader, val_loader, device, lr=0.001,
          num_epochs=5, weight_decay=0.0,patience=3, use_early_stopping=True, 
          save_path:str="trained_models"):
    os.makedirs(save_path, exist_ok=True)
    criterion = nn.BCELoss()

    #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Add L2 regualarization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    best_val_loss = float('inf')
    patience_counter = 0

    outer_bar = trange(num_epochs, desc="Epochs")
    train_losses = []
    val_losses = []

    for epoch in outer_bar:
        model.train()
        total_loss = 0

        inner_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        for users, items, labels in inner_bar:
            users, items, labels = users.to(device), items.to(device), labels.to(device)
            preds = model(users, items)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            inner_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Validating Epoch {epoch+1}", leave=False)
        with torch.no_grad():
            for users, items, labels in val_bar:
                users, items, labels = users.to(device), items.to(device), labels.to(device)
                preds = model(users, items)
                loss = criterion(preds, labels)
                val_loss += loss.item()
                val_bar.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_val_loss:.4f}")


        outer_bar.set_description(f"Epoch {epoch+1}")
        outer_bar.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss)

        # Early stopping logic (optional)
        if use_early_stopping:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pth")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("\nEarly stopping triggered.")
                    break
            
     # Final save: only if early stopping is off
    if not use_early_stopping:
        torch.save(model.state_dict(), os.path.join(save_path, "model.pth"))
        print("Final model saved as model.pth")

    return train_losses, val_losses

if __name__ == "__main__":
    # Preprocess data
    data_path = "data_raw/ratings.dat"
    output_path = "data_preprocessed"
    models_path = "trained_models"
    full_df = process.preprocess(data_path=data_path, output_path=output_path, save=False)

    # Split
    train_df, temp_df = train_test_split(full_df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    # Dataloaders
    train_loader, num_users, num_items = dataloader.get_dataloader_from_df(train_df, batch_size=256, shuffle=True)
    val_loader, _, _ = dataloader.get_dataloader_from_df(val_df, batch_size=256, shuffle=False)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NCF(num_users=num_users, num_items=num_items).to(device)

    # Train
    train(model, train_loader, val_loader, device)
