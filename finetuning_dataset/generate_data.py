import os
import json
import openml
from finetuning_dataset.utils import (
    MODEL_NAME_MAP, MODEL_BENCHMARK_MAP, DATASET_MAP, HYPERPARAM_BOUNDS
)
from bayes_opt import BayesianOptimization

DATA_DIR = "valid_data_json"


def hyps_to_int(config: dict, model_name: str):
    """Round hyperparameters to ints."""
    rounding_keys = {
        "rf": ["max_depth", "min_samples_split", "min_samples_leaf"],
        "nn": ["depth", "width", "batch_size"],
        "xgb": ["max_depth"]
    }
    for key in rounding_keys.get(model_name, []):
        config[key] = round(config[key])


def evaluate_metrics(benchmark, config, model_name):
    hyps_to_int(config, model_name)
    benchmark_info = benchmark.objective_function(config)["info"]
    val_scores = benchmark_info.get("val_scores", {})
    return {
        "accuracy": val_scores.get("acc"),
        "balanced_accuracy": val_scores.get("balanced_accuracy"),
        "f1_score": val_scores.get("f1"),
        "precision": val_scores.get("precision")
    }


def generate_training_data(model_name, task_id, dataset_name, num_expts=3, n_trials=25, n_initial_points=5):
    # Initialize benchmark for the specific model and dataset
    benchmark_class = MODEL_BENCHMARK_MAP[model_name]
    benchmark = benchmark_class(task_id=task_id)
    # Define search space based on model
    pbounds = HYPERPARAM_BOUNDS[model_name]

    for j in range(num_expts):
        os.makedirs(f"{DATA_DIR}/{dataset_name}/{model_name}/exp0{j+1}", exist_ok=True)

        bo = BayesianOptimization(
            f=lambda **params: benchmark.objective_function({
                **params,
                **({
                    'max_depth': round(params['max_depth']),
                    'min_samples_split': round(params['min_samples_split']),
                    'min_samples_leaf': round(params['min_samples_leaf'])
                } if model_name == 'rf' else {
                    'depth': round(params['depth']),
                    'width': round(params['width']),
                    'batch_size': round(params['batch_size'])
                } if model_name == 'nn' else {
                    'max_depth': round(params['max_depth'])
                })
            })["info"]["val_scores"]["acc"],
            pbounds=pbounds,
            verbose=2,
        )

        # Run Bayesian Optimization with initial points and trials
        bo.maximize(init_points=n_initial_points, n_iter=n_trials)

        optimization_history = []
        # Initial steps handled differently to future trials
        for i in range(n_initial_points):
            trial = bo.res[i]
            config = trial["params"]
            hyps_to_int(config, model_name)
            result = trial["target"]

            metrics = evaluate_metrics(benchmark, config, model_name)

            optimization_history.append({
                "hyperparameters": config,
                "performance": metrics,
            })

            step_data = {
                "model_name": MODEL_NAME_MAP[model_name],
                "task": "Classification",  # hpo_bench datasets are all classification
                "metric": "accuracy",
                "current_step": 0,
                "optimization_history": [], # empty for initial steps
                "current_hyperparameters": config,
                "current_performance": optimization_history[i]["performance"],
            }
            with open(f"{DATA_DIR}/{dataset_name}/{model_name}/exp0{j+1}/initial_step_{i}.json", "w") as f:
                json.dump(step_data, f, indent=4)

        # Save jsons for steps after initial points
        for i in range(n_initial_points, len(bo.res)):
            trial = bo.res[i]
            config = trial["params"]
            hyps_to_int(config, model_name)
            result = trial["target"]

            metrics = evaluate_metrics(benchmark, config, model_name)

            optimization_history.append({
                "hyperparameters": config,
                "performance": metrics,
            })
            step_num = i + 1 - n_initial_points
            step_data = {
                "model_name": MODEL_NAME_MAP[model_name],
                "task": "Classification",
                "metric": "accuracy",
                "current_step": step_num,
                "optimization_history": optimization_history[:i],  # Include all previous steps up to current
                "current_hyperparameters": config,
                "current_performance": optimization_history[i]["performance"],
            }

            # Save generated data to JSON file for each step
            with open(f"{DATA_DIR}/{dataset_name}/{model_name}/exp0{j+1}/step_{step_num}.json", "w") as f:
                json.dump(step_data, f, indent=4)


if __name__ == "__main__":
    for dataset_name, task_id in DATASET_MAP.items():
        for model in MODEL_BENCHMARK_MAP.keys():
            os.makedirs(f"{DATA_DIR}/{dataset_name}/{model}", exist_ok=True)
            generate_training_data(model, task_id, dataset_name)
