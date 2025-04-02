import json
import os
from main import run  # Assumes your main file is named main.py and contains the run() function

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

        # Experiment name
        exp_name = f"ablation_{param_name}_{val}"

        # Save config for this run 
        temp_config_path = f"temp_config_{param_name}_{val}.json"
        with open(temp_config_path, "w") as f:
            json.dump(config, f, indent=4)

        # Run experiment
        temp_train_loss, temp_val_loss = run(
            data_path=data_path,
            output_path=output_path,
            config_path=temp_config_path,
            results_path=results_path,
            save_path=save_path,
            exp_name=exp_name,
            ablation=True
        )
        temp_result = {
            "train_loss": temp_train_loss,
            "val_loss": temp_val_loss
        }


        all_results[val] = temp_result

        # Clean up temp config
        os.remove(temp_config_path)

    # Save all ablation results
    os.makedirs(results_path, exist_ok=True)
    output_file_path = os.path.join(results_path, f"ablation_{param_name}.json")
    with open(output_file_path, "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(f" Ablation results saved to {output_file_path}")



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
        param_values=[64,128],
        data_path=DATA_PATH, 
        output_path=OUTPUT_PATH, 
        config_path= CONFIG_PATH,
        results_path=RESULTS_PATH, 
        save_path=MODEL_PATH
    ) 