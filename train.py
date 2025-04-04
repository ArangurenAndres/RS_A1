import torch
import os
import torch.nn as nn
from tqdm import trange, tqdm
import torch
import os
import torch.nn as nn
from tqdm import trange, tqdm

def train(model, train_loader, device, lr=0.001,
          num_epochs=5, weight_decay=0.0,
          save_path: str = "trained_models", model_name="model.pth"):
    
    os.makedirs(save_path, exist_ok=True)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    outer_bar = trange(num_epochs, desc="Epochs")
    train_losses = []

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

        outer_bar.set_description(f"Epoch {epoch+1}")
        outer_bar.set_postfix(train_loss=avg_train_loss)

    # Save the final model
    model.eval()
    torch.save(model.state_dict(), os.path.join(save_path, model_name))
    print(f"Final model saved as {model_name}")

    return train_losses


