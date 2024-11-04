import os
import json
import csv
from finetuning_dataset.utils import HYPERPARAM_BOUNDS


def extract_discriminative_prompt(json_data, dataset_info, model_tag):
    model_name = json_data["model_name"]
    task = json_data["task"]
    metric = json_data["metric"]

    if "optimization_history" in json_data and json_data["optimization_history"]:
        prompt = (
            f"The following are examples of the performance of a {model_name} measured in {metric} and the corresponding model hyperparameter configurations. "
            f"The model is evaluated on a tabular {task} task containing {dataset_info['num_classes']} classes. "
            f"The tabular dataset contains {dataset_info['num_samples']} samples and {dataset_info['num_features']} features "
            f"({dataset_info['num_categorical_features']} categorical, {dataset_info['num_continuous_features']} numerical). "
            f"Your response should only contain the predicted accuracy in the format ## performance ##.\n"
        )

        for history in json_data["optimization_history"]:
            hyperparams = ", ".join([f"{k}: {v}" for k, v in history["hyperparameters"].items()])
            performance = history["performance"]["accuracy"]
            prompt += f"Hyperparameter configuration: {hyperparams}\nPerformance: {performance}\n"

        current_hyperparams = ", ".join([f"{k}: {v}" for k, v in json_data["current_hyperparameters"].items()])
        prompt += f"Hyperparameter configuration: {current_hyperparams}\nPerformance:"

        response = f"## {json_data['current_performance']['accuracy']} ##"
        return prompt, response
    else:
        return None, None


def extract_candidate_sampling_prompt(json_data, dataset_info, model_tag):
    model_name = json_data["model_name"]
    task = json_data["task"]
    metric = json_data["metric"]
    hyperparam_bounds = HYPERPARAM_BOUNDS.get(model_tag)

    if "optimization_history" in json_data and json_data["optimization_history"] and hyperparam_bounds:
        prompt = (
            f"The following are examples of the performance of a {model_name} measured in {metric} and the corresponding model hyperparameter configurations. "
            f"The model is evaluated on a tabular {task} task containing {dataset_info['num_classes']} classes. "
            f"The tabular dataset contains {dataset_info['num_samples']} samples and {dataset_info['num_features']} features "
            f"({dataset_info['num_categorical_features']} categorical, {dataset_info['num_continuous_features']} numerical). "
            f"The allowable ranges for the hyperparameters are: {hyperparam_bounds}. "
            f"Recommend a configuration that can achieve the target performance of {json_data['current_performance']['accuracy']}. "
            f"Do not recommend values at the minimum or maximum of allowable range, do not recommend rounded values. "
            f"Recommend values with the highest possible precision, as requested by the allowed ranges. Your response must only contain the predicted configuration, in the format ## configuration ##.\n"
        )

        for history in json_data["optimization_history"]:
            performance = history["performance"]["accuracy"]
            hyperparams = ", ".join([f"{k}: {v}" for k, v in history["hyperparameters"].items()])
            prompt += f"Performance: {performance}\nHyperparameter configuration: {hyperparams}\n"

        prompt += f"Performance: {json_data['current_performance']['accuracy']}\nHyperparameter configuration:"
        current_hyperparams = ", ".join([f"{k}: {v}" for k, v in json_data["current_hyperparameters"].items()])
        response = f"## {current_hyperparams} ##"
        return prompt, response
    else:
        return None, None


def generate_prompts(data_dir, dataset_info_dir, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for dataset_name in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        # Load dataset info from the respective JSON file
        dataset_info_path = os.path.join(dataset_info_dir, dataset_name, "dataset_info.json")
        if not os.path.exists(dataset_info_path):
            print(f"Dataset info for {dataset_name} not found. Skipping.")
            continue

        with open(dataset_info_path, "r") as f:
            dataset_info = json.load(f)

        for model_name in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model_name)
            if not os.path.isdir(model_path):
                continue

            output_file_discriminative = os.path.join(output_folder, f"{dataset_name}_{model_name}_discriminative_training_data.csv")
            output_file_candidate = os.path.join(output_folder, f"{dataset_name}_{model_name}_candidate_training_data.csv")

            with open(output_file_discriminative, "w", newline="") as out_f_disc, open(output_file_candidate, "w", newline="") as out_f_cand:
                csv_writer_disc = csv.writer(out_f_disc)
                csv_writer_cand = csv.writer(out_f_cand)
                csv_writer_disc.writerow(["prompt", "response"])
                csv_writer_cand.writerow(["prompt", "response"])

                for exp_name in os.listdir(model_path):
                    exp_path = os.path.join(model_path, exp_name)
                    if not os.path.isdir(exp_path):
                        continue

                    for json_file in os.listdir(exp_path):
                        if json_file.endswith(".json") and not json_file.startswith("initial_step_"):
                            json_path = os.path.join(exp_path, json_file)
                            with open(json_path, "r") as f:
                                json_data = json.load(f)

                                # Generate discriminative surrogate model prompts
                                prompt_disc, response_disc = extract_discriminative_prompt(json_data, dataset_info, model_name)
                                if prompt_disc and response_disc:
                                    csv_writer_disc.writerow([prompt_disc, response_disc])

                                # Generate candidate sampling prompts
                                prompt_cand, response_cand = extract_candidate_sampling_prompt(json_data, dataset_info, model_name)
                                if prompt_cand and response_cand:
                                    csv_writer_cand.writerow([prompt_cand, response_cand])

                print(f"Discriminative prompts saved to {output_file_discriminative}")
                print(f"Candidate sampling prompts saved to {output_file_candidate}")

if __name__ == "__main__":
    data_dir = "valid_data_json"
    dataset_info_dir = "dataset_info"
    output_folder = "valid_data_csv"
    generate_prompts(data_dir, dataset_info_dir, output_folder)
