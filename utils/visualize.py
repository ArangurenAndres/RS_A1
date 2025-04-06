import matplotlib.pyplot as plt
import seaborn as sns
import json

def plot_loss_curve_from_file(results_path="results/loss_results.json", title="Training and Validation Loss"):
    """
    Loads training and validation loss from a JSON file and plots them using seaborn.

    Args:
        results_path (str): Path to the JSON file containing "train_loss" and "val_loss".
        title (str): Title of the plot.
    """
    # Load JSON
    with open(results_path, 'r') as f:
        results = json.load(f)

    train_losses = results["train_loss"]
    val_losses = results.get("val_loss", None)  # Optional in case val_loss is missing
    epochs = list(range(1, len(train_losses) + 1))

    # Plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=epochs, y=train_losses, label="Train Loss", marker="o")
    
    if val_losses is not None:
        sns.lineplot(x=epochs, y=val_losses, label="Validation Loss", marker="s")

    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results_path = "results/exp_test.json"
    plot_loss_curve_from_file(results_path)
