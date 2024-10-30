from hpobench.benchmarks.ml import RandomForestBenchmark, NNBenchmark
from finetuning_dataset.xgb_benchmark import CustomXGBoostBenchmark

MODEL_NAME_MAP = {
    "rf": "Random Forest",
    "nn": "Neural Network (Multi-Layer Perceptron)",
    "xgb": "XGBoost"
}

MODEL_BENCHMARK_MAP = {
    "rf": RandomForestBenchmark,
    "nn": NNBenchmark,
    "xgb": CustomXGBoostBenchmark,
}

DATASET_MAP = {
    "credit_g": 31,
    "vehicle": 53,
    "kc1": 3917,
    "phoneme": 9952,
    "blood_transfusion": 10101,
    "australian": 146818,
    "car": 146821,
    "segment": 146822,
}

HYPERPARAM_BOUNDS = {
    "rf": {"max_depth": (1, 50), "min_samples_split": (2, 128), "max_features": (0, 1.0), "min_samples_leaf": (1, 20)},
    "nn": {"depth": (1, 3), "width": (16, 1024), "batch_size": (4, 256), "alpha": (1e-8, 1.0), "learning_rate_init": (1e-5, 1.0)},
    "xgb": {"eta": (2**-10, 1.0), "max_depth": (1, 50), "colsample_bytree": (0.1, 1.0), "reg_lambda": (2**-10, 2**10)},
}