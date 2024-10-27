import os
import json
import openml
from hpobench.benchmarks.ml import RandomForestBenchmark, NNBenchmark, XGBoostBenchmark
from bayes_opt import BayesianOptimization


MODEL_NAME_MAP = {
    "rf": "Random Forest",
    "nn": "Neural Network (Multi-Layer Perceptron)",
    "xgb": "XGBoost"
}

MODEL_BENCHMARK_MAP = {
    "rf": RandomForestBenchmark,
    "nn": NNBenchmark,
    "xgb": XGBoostBenchmark,
}

DATASET_MAP = {
    "credit_g": 31,
    # "vehicle": 53,
    # "kc1": 3917,
    # "phoneme": 9952,
    # "blood_transfusion": 10101,
    # "australian": 146818,
    # "car": 146821,
    # "segment": 146822,
}

# Define hyperparameter bounds for each model
HYPERPARAM_BOUNDS = {
    "rf": {"max_depth": (1, 50), "min_samples_split": (2, 128), "max_features": (0, 1.0), "min_samples_leaf": (1, 20)},
    "nn": {"depth": (1, 3), "width": (16, 1024), "batch_size": (4, 256), "alpha": (1e-8, 1.0), "learning_rate_init": (1e-5, 1.0)},
    "xgb": {"eta": (2**-10, 1.0), "max_depth": (1, 50), "colsample_bytree": (0.1, 1.0), "reg_lambda": (2**-10, 2**10)},
}

# Directory to save training data
data_dir = "training_data"
os.makedirs(data_dir, exist_ok=True)


def get_dataset_info(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    labels = dataset.retrieve_class_labels()
    X, _, categorical_mask, _= dataset.get_data()

    num_features = X.shape[1]
    num_categorical_features = sum(categorical_mask)
    num_continuous_features = num_features - num_categorical_features

    return {
        "dataset_name": dataset.name,
        "num_classes": len(labels) if labels else 0,
        "num_samples": X.shape[0],
        "num_features": num_features,
        "num_categorical_features": num_categorical_features,
        "num_continuous_features": num_continuous_features,
    }


def get_metrics(benchmark, config, model_name):
    if model_name == "rf":
        benchmark_info = benchmark.objective_function({
            **config,
            'max_depth': round(config['max_depth']),
            'min_samples_split': round(config['min_samples_split']),
            'min_samples_leaf': round(config['min_samples_leaf'])
        })["info"]
    elif model_name == "nn":
        benchmark_info = benchmark.objective_function({
            **config,
            'depth': round(config['depth']),
            'width': round(config['width']),
            'batch_size': round(config['batch_size'])
        })["info"]
    elif model_name == "xgb":
        benchmark_info = benchmark.objective_function({
            **config,
            'max_depth': round(config['max_depth'])
        })["info"]
    else:
        benchmark_info = benchmark.objective_function(config)["info"]
    
    balanced_accuracy = benchmark_info["val_scores"].get("balanced_accuracy", None)
    f1 = benchmark_info["val_scores"].get("f1", None)
    precision = benchmark_info["val_scores"].get("precision", None)
    return balanced_accuracy, f1, precision


def hyps_to_int(config: dict, model_name: str):
    if model_name == "rf":
        config['max_depth'] = round(config['max_depth'])
        config['min_samples_split'] = round(config['min_samples_split'])
        config['min_samples_leaf'] = round(config['min_samples_leaf'])
    elif model_name == "nn":
        config['depth'] = round(config['depth'])
        config['width'] = round(config['width'])
        config['batch_size'] = round(config['batch_size'])
    elif model_name == "xgb":
        config['max_depth'] = round(config['max_depth'])


def generate_training_data(model_name, dataset_id, dataset_name, num_expts=5, n_trials=25, n_initial_points=5):
    # Get dataset information from OpenML
    dataset_info = get_dataset_info(dataset_id)

    # Initialize benchmark for the specific model and dataset
    benchmark_class = MODEL_BENCHMARK_MAP[model_name]
    benchmark = benchmark_class(task_id=dataset_id)
    # Define search space based on model
    pbounds = HYPERPARAM_BOUNDS[model_name]

    for j in range(num_expts):
        os.makedirs(f"{data_dir}/{dataset_name}/{model_name}/exp0{j+1}", exist_ok=True)

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

            bal_acc, f1, precision = get_metrics(benchmark, config, model_name)

            optimization_history.append({
                "hyperparameters": config,
                "performance": {
                    "accuracy": result,
                    "balanced_accuracy": bal_acc,
                    "f1_score": f1,
                    "precision": precision
                }
            })

            step_data = {
                "model_name": MODEL_NAME_MAP[model_name],
                "task": "Classification",  # hpo_bench datasets are all classification
                "metric": "accuracy",
                "dataset_info": dataset_info,
                "current_step": 0,
                "optimization_history": [], # empty for initial steps
                "current_hyperparameters": config,
                "current_performance": optimization_history[i]["performance"],
            }
            with open(f"{data_dir}/{dataset_name}/{model_name}/exp0{j+1}/initial_step_{i}.json", "w") as f:
                json.dump(step_data, f, indent=4)

        # Save jsons for steps after initial points
        for i in range(n_initial_points, len(bo.res)):
            trial = bo.res[i]
            config = trial["params"]
            hyps_to_int(config, model_name)
            result = trial["target"]

            bal_acc, f1, precision = get_metrics(benchmark, config, model_name)

            optimization_history.append({
                "hyperparameters": config,
                "performance": {
                    "accuracy": result,
                    "balanced_accuracy": bal_acc,
                    "f1_score": f1,
                    "precision": precision
                }
            })
            step_num = i + 1 - n_initial_points
            step_data = {
                "model_name": MODEL_NAME_MAP[model_name],
                "task": "Classification",
                "metric": "accuracy",
                "dataset_info": dataset_info,
                "current_step": step_num,
                "optimization_history": optimization_history[:i],  # Include all previous steps up to current
                "current_hyperparameters": config,
                "current_performance": optimization_history[i]["performance"],
            }

            # Save generated data to JSON file for each step
            with open(f"{data_dir}/{dataset_name}/{model_name}/exp0{j+1}/step_{step_num}.json", "w") as f:
                json.dump(step_data, f, indent=4)

# Run for all models and datasets
for model in MODEL_BENCHMARK_MAP.keys():
    for dataset_name, dataset_id in DATASET_MAP.items():
        os.makedirs(f"{data_dir}/{dataset_name}/{model}", exist_ok=True)
        generate_training_data(model, dataset_id, dataset_name)
