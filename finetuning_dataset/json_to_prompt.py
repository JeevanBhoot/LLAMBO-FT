import os
import json
import csv

def extract_prompts_from_json(json_data, dataset_name, model_name):
    dataset_info = json_data['dataset_info']
    model_name = json_data['model_name']
    task = json_data['task']
    metric = json_data['metric']

    if 'optimization_history' in json_data and json_data['optimization_history']:
        prompt = (
            f"The following are examples of the performance of a {model_name} measured in {metric} and the corresponding model hyperparameter configurations. "
            f"The model is evaluated on a tabular {task} task containing {dataset_info['num_classes']} classes. "
            f"The tabular dataset contains {dataset_info['num_samples']} samples and {dataset_info['num_features']} features "
            f"({dataset_info['num_categorical_features']} categorical, {dataset_info['num_continuous_features']} numerical). "
            f"Your response should only contain the predicted accuracy in the format ## performance ##.\n"
        )

        for history in json_data['optimization_history']:
            hyperparams = ', '.join([f"{k}: {v}" for k, v in history['hyperparameters'].items()])
            performance = history['performance']['accuracy']
            prompt += f"Hyperparameter configuration: {hyperparams}\nPerformance: {performance}\n"

        current_hyperparams = ', '.join([f"{k}: {v}" for k, v in json_data['current_hyperparameters'].items()])
        prompt += f"Hyperparameter configuration: {current_hyperparams}\nPerformance:"

        response = f"## {json_data['current_performance']['accuracy']} ##"
        return prompt, response
    else:
        return None, None

def generate_prompts(data_dir, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    for dataset_name in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue

        for model_name in os.listdir(dataset_path):
            model_path = os.path.join(dataset_path, model_name)
            if not os.path.isdir(model_path):
                continue

            output_file = os.path.join(output_folder, f"{dataset_name}_{model_name}_training_data.csv")
            with open(output_file, 'w', newline='') as out_f:
                csv_writer = csv.writer(out_f)
                csv_writer.writerow(["prompt", "response"])

                for exp_name in os.listdir(model_path):
                    exp_path = os.path.join(model_path, exp_name)
                    if not os.path.isdir(exp_path):
                        continue

                    for json_file in os.listdir(exp_path):
                        if json_file.endswith('.json') and not json_file.startswith('initial_step_'):
                            json_path = os.path.join(exp_path, json_file)
                            with open(json_path, 'r') as f:
                                json_data = json.load(f)
                                prompt, response = extract_prompts_from_json(json_data, dataset_name, model_name)
                                if prompt and response:
                                    csv_writer.writerow([prompt, response])

                print(f"Prompts saved to {output_file}")

data_dir = "training_data"
output_folder = "csv_output"
generate_prompts(data_dir, output_folder)
