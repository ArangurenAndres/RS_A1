import json
import matplotlib.pyplot as plt
import seaborn as sns

def plot_ablation_results(result_path, param_name, plot_val=True):
    # Load results
    with open(result_path, "r") as f:
        results = json.load(f)

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    for val, losses in results.items():
        label = f"{param_name}={val}"
        y = losses["train_loss"] if plot_val else losses["train_loss"]
        x = list(range(1, len(y) + 1))
        plt.plot(x, y, label=label)

    metric = "Training Loss" if plot_val else "Training Loss"
    plt.title(f"Ablation Study: {metric} vs {param_name}")
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend(title=param_name)
    plt.tight_layout()

    # Save and show
    output_fig = result_path.replace(".json", f"_{'val' if plot_val else 'train'}_plot.png")
    plt.savefig(output_fig, dpi=300)
    print(f"✅ Plot saved to {output_fig}")
    plt.show()


if __name__ == "__main__":
    # Example usage
    plot_ablation_results(
        result_path="ablation_results/ablation_batch_size.json",
        param_name="batch_size",
        plot_val=True  # Set to False if you want training loss
    ) for now dont save the figurea and also p[lot the training loss not the validation loss