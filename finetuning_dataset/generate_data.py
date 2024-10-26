import os
import json
import openml
from hpobench.benchmarks.ml import RandomForestBenchmark, NNBenchmark, XGBoostBenchmark
from bayes_opt import BayesianOptimization

# Map for models and their respective HPOBench classes
MODEL_NAME_MAP = {
    "rf": "Random Forest",
    "nn": "Neural Network (Multi-Layer Perceptron)",
    "xgb": "XGBoost"
}

MODEL_BENCHMARK_MAP = {
    "rf": RandomForestBenchmark,
    # "nn": NNBenchmark,
    # "xgb": XGBoostBenchmark,
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

# Define hyperparameter bounds for each model (example bounds for simplicity)
HYPERPARAM_BOUNDS = {
    "rf": {"max_depth": (1, 50), "min_samples_split": (2, 128), "max_features": (0, 1.0), "min_samples_leaf": (1, 20)},
    # "nn": {"learning_rate": (0.0001, 0.1), "batch_size": (16, 128), "n_units": (10, 100)},
    # "xgb": {"max_depth": (1, 10), "n_estimators": (10, 200), "learning_rate": (0.01, 0.3)}
}

# Directory to save training data
save_res_dir = "training_data"
os.makedirs(save_res_dir, exist_ok=True)

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

def generate_training_data(model_name, dataset_id, n_trials=25, n_initial_points=5):
    # Get dataset information from OpenML
    dataset_info = get_dataset_info(dataset_id)

    # Initialize benchmark for the specific model and dataset
    benchmark_class = MODEL_BENCHMARK_MAP[model_name]
    benchmark = benchmark_class(task_id=dataset_id)
    # Define search space based on model
    pbounds = HYPERPARAM_BOUNDS[model_name]

    # Define Bayesian Optimization process
    bo = BayesianOptimization(
        f=lambda **params: benchmark.objective_function(
            {
                **params, 
                "max_depth": int(params["max_depth"]),
                "min_samples_split": int(params["min_samples_split"]),
                "min_samples_leaf": int(params["min_samples_leaf"])
            }
        )["info"]["val_scores"]["acc"],
        pbounds=pbounds,
        verbose=2,
    )

    # Run Bayesian Optimization with initial points and trials
    bo.maximize(init_points=n_initial_points, n_iter=n_trials)

    # Collect and format data for full context
    optimization_history = []
    for trial in bo.res:
        config = trial["params"]
        result = trial["target"]
        optimization_history.append({"hyperparameters": config, "performance": result})



    # Save detailed optimization history for each step
    for i, trial in enumerate(bo.res):
        config = trial["params"]
        result = trial["target"]

        # Combine all information in a JSON file
        step_data = {
            "model_name": MODEL_NAME_MAP[model_name],
            "task": "Classification",
            "metric": "accuracy",
            "dataset_info": dataset_info,
            "current_step": i,
            "optimization_history": optimization_history[:i],  # Include all previous steps up to current
            "current_hyperparameters": config,
            "current_performance": result,
        }

        # Save generated data to JSON file for each step
        with open(f"{save_res_dir}/{model_name}_{dataset_id}_step_{i}.json", "w") as f:
            json.dump(step_data, f, indent=4)

# Run for all models and datasets
for model in MODEL_BENCHMARK_MAP.keys():
    for dataset_name, dataset_id in DATASET_MAP.items():
        generate_training_data(model, dataset_id)
