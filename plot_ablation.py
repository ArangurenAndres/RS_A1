import json
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ablation_results(result_path, param_name):
    # Load results
    with open(result_path, "r") as f:
        results = json.load(f)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    for val, losses in results.items():
        label = f"{param_name}={val}"
        y = losses["train_loss"]  # Only training loss
        x = list(range(1, len(y) + 1))
        plt.plot(x, y, label=label)

    plt.title(f"Ablation Study: Training Loss vs {param_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.legend(title=param_name)
    plt.tight_layout()

    plt.show()  # Only show the figure


if __name__ == "__main__":
    plot_ablation_results(
        result_path="ablation_results/ablation_embedding_dim.json",
        param_name="Embedding Dimension"
    )
