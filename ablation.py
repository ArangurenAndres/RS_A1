import json
import os
from main import run  # Assumes run() is defined in main.py

def run_ablation_study(param_name, param_values,
                       data_path, output_path, config_path,
                       results_path, save_path):
    with open(config_path, "r") as f:
        base_config = json.load(f)

    all_results = {}

    for val in param_values:
        print(f"Running ablation for {param_name} = {val}")
        config = base_config.copy()
        config[param_name] = val

        # Create experiment name
        exp_name = f"ablation_{param_name}_{val}"

        # Save temporary config for this run
        temp_config_path = f"temp_config_{param_name}_{val}.json"
        with open(temp_config_path, "w") as f:
            json.dump(config, f, indent=4)

        # Run training
        train_loss = run(
            data_path=data_path,
            output_path=output_path,
            config_path=temp_config_path,
            results_path=results_path,
            save_path=save_path,
            exp_name=exp_name,
            ablation=True,  # avoid writing default results file
            model_name=f"model_{exp_name}.pth"
        )

        # Record result
        all_results[str(val)] = {
            "train_loss": train_loss
        }

        # Clean up
        os.remove(temp_config_path)

    # Save ablation results
    os.makedirs(results_path, exist_ok=True)
    output_file_path = os.path.join(results_path, f"ablation_{param_name}.json")
    with open(output_file_path, "w") as f:
        json.dump(all_results, f, indent=4)

    print(f"Ablation results saved to {output_file_path}")



if __name__ == "__main__":
    # Paths
    DATA_PATH = "data_raw/ratings.dat"
    OUTPUT_PATH = "data_preprocessed"
    CONFIG_PATH = "config.json"
    RESULTS_PATH = "ablation_results"
    MODEL_PATH = "trained_models"

    # Choose your ablation target
    run_ablation_study(
        param_name="batch_size", 
        param_values=[64,128,256,512,1024],
        data_path=DATA_PATH, 
        output_path=OUTPUT_PATH, 
        config_path= CONFIG_PATH,
        results_path=RESULTS_PATH, 
        save_path=MODEL_PATH
    ) 