import os
import json
import csv
from finetuning_dataset.utils import HYPERPARAM_BOUNDS


def extract_warmstarting_prompts(dataset_info, model_tag, exp_path):
    """
    Generates warm-starting prompts for Bayesian Optimization in AutoML, each with and without statistical information.

    Parameters:
    - dataset_info (dict): Dictionary containing dataset statistics.
    - model_tag (str): Identifier for the model to retrieve hyperparameter bounds.
    - exp_path (str): Path to the directory containing initial step JSON files.

    Returns:
    - warmstarting_prompts (list): A list of tuples, each containing a prompt and corresponding recommendations.
    """
    warmstarting_prompts = []
    hyperparam_bounds = HYPERPARAM_BOUNDS.get(model_tag)

    for i in range(5):
        recommendations = []
        for j in range(i + 1):
            initial_step_file = os.path.join(exp_path, f"initial_step_{j}.json")
            if not os.path.exists(initial_step_file):
                continue
            with open(initial_step_file, "r") as f:
                json_data = json.load(f)
                recommendations.append(json_data["current_hyperparameters"])

        if not recommendations:
            continue

        # Extract model and task information from the JSON data
        model_name = json_data["model_name"]
        task = json_data["task"]
        metric = json_data["metric"]

        # Construct statistical information
        statistical_information = (
            f"Considering one-hot encoding for categorical features, the total number of input features for the "
            f"{model_name} is {dataset_info['total_features_after_encoding']}. We are standardizing numerical values to have mean 0 and std 1. "
            f"The skewness of each feature is {dataset_info['skewness']}. The number of features that have strong correlation "
            f"(defined as > 0.5 or < -0.5) with the target feature is {dataset_info['strong_correlations_with_target']}. "
            f"Of the {dataset_info['num_pairwise_relationships']} pairwise feature relationships, "
            f"{dataset_info['strong_pairwise_correlations']} pairs of features are strongly correlated (> 0.5 or < -0.5)."
        )

        # Base prompt without statistical information
        base_prompt = (
            f"You are assisting me with automated machine learning using {model_name} for a {task} task. "
            f"The {task} performance is measured using {metric}. The dataset has {dataset_info['num_samples']} samples "
            f"with {dataset_info['num_features']} total features, of which {dataset_info['num_continuous_features']} are numerical "
            f"and {dataset_info['num_categorical_features']} are categorical. Class distribution is {dataset_info['class_distribution']}. "
        )

        # Common ending for both prompts
        prompt_ending = (
            f"I'm exploring a subset of hyperparameters detailed as: {hyperparam_bounds}. "
            f"Please suggest {i + 1} diverse yet effective configurations to initiate a Bayesian Optimization process for hyperparameter tuning. "
            "You mustn't include 'None' in the configurations. Your response should include only a list of dictionaries, "
            "where each dictionary describes one recommended configuration. Do not enumerate the dictionaries."
        )

        # Prompt with statistical information
        prompt_with_stats = f"{base_prompt}{statistical_information} {prompt_ending}"

        # Prompt without statistical information
        prompt_without_stats = f"{base_prompt}{prompt_ending}"

        # Append both prompts to the list with the same recommendations
        warmstarting_prompts.append((prompt_with_stats, recommendations))
        warmstarting_prompts.append((prompt_without_stats, recommendations))

    return warmstarting_prompts

def main():
    dataset_info_path = "dataset_info"
    training_data_path = "training_data_new"
    output_folder = "csv_output_X"
    os.makedirs(output_folder, exist_ok=True)

    # Iterate through each dataset in the training_data_new directory
    for dataset_name in os.listdir(training_data_path):
        dataset_dir = os.path.join(training_data_path, dataset_name)
        if not os.path.isdir(dataset_dir):
            continue

        # Load the corresponding dataset_info.json file
        dataset_info_file = os.path.join(dataset_info_path, dataset_name, "dataset_info.json")
        if not os.path.exists(dataset_info_file):
            print(f"Dataset info file not found for {dataset_name}, skipping...")
            continue

        with open(dataset_info_file, "r") as f:
            dataset_info = json.load(f)

        # Iterate through each model (rf, nn, xgb) in the dataset directory
        for model_tag in os.listdir(dataset_dir):
            model_dir = os.path.join(dataset_dir, model_tag)
            if not os.path.isdir(model_dir):
                continue

            # Define unique CSV file for each (dataset, model) pair
            output_csv_file = os.path.join(output_folder, f"{dataset_name}_{model_tag}_warmstarting.csv")

            # Open CSV file for writing prompts and recommendations
            with open(output_csv_file, mode="w", newline="") as csv_file:
                fieldnames = ["prompt", "response"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                # Iterate through each experiment (exp01 to exp05) in the model directory
                for exp_name in os.listdir(model_dir):
                    exp_dir = os.path.join(model_dir, exp_name)
                    if not os.path.isdir(exp_dir):
                        continue

                    # Generate warmstarting prompts
                    warmstarting_prompts = extract_warmstarting_prompts(dataset_info, model_tag, exp_dir)

                    # Write the prompts and recommendations to the CSV file
                    for prompt, recommendations in warmstarting_prompts:
                        writer.writerow({
                            "prompt": prompt,
                            "response": recommendations,
                        })

if __name__ == "__main__":
    main()