import torch
import os
import torch.nn as nn
from tqdm import trange, tqdm

def train(model, train_loader, val_loader, device,
          lr=0.001, num_epochs=5, weight_decay=0.0,
          save_path: str = "trained_models", model_name="model.pth"):
    
    os.makedirs(save_path, exist_ok=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    outer_bar = trange(num_epochs, desc="Epochs")
    train_losses = []
    val_losses = []

    for epoch in outer_bar:
        model.train()
        total_train_loss = 0

        inner_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}", leave=False)
        for users, items, labels in inner_bar:
            users, items, labels = users.to(device), items.to(device), labels.to(device)

            preds = model(users, items)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            inner_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for users, items, labels in val_loader:
                users, items, labels = users.to(device), items.to(device), labels.to(device)
                preds = model(users, items)
                loss = criterion(preds, labels)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        outer_bar.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss)

    # Save the final model
    torch.save(model.state_dict(), os.path.join(save_path, model_name))
    print(f"Final model saved as {model_name}")

    return train_losses, val_losses
