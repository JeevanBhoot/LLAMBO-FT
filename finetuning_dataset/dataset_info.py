import os
import json
import openml
import numpy as np
import pandas as pd
import argparse
from collections import Counter
from finetuning_dataset.utils import DATASET_MAP, DATASET_OUT_DOMAIN_MAP


def parse_args():
    parser = argparse.ArgumentParser(description="Script for generating dataset information.")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default="dataset_info_out_domain", 
        help="Directory to store the generated dataset information."
    )
    return parser.parse_args()


def get_dataset_info(task_id):
    task = openml.tasks.get_task(task_id)
    dataset = openml.datasets.get_dataset(task.dataset_id)
    target = dataset.default_target_attribute
    labels = dataset.retrieve_class_labels(target)
    X, y, categorical_mask, _ = dataset.get_data(target=target)

    # Convert labels to numeric if necessary
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = y.astype('category').cat.codes

    num_features = X.shape[1]
    num_categorical_features = sum(categorical_mask)
    num_continuous_features = num_features - num_categorical_features

    # Calculate class distribution
    class_counts = Counter(y)
    class_distribution = {labels[i]: count for i, count in class_counts.items()}

    # One-hot encoding for categorical features
    X_encoded = pd.get_dummies(X, columns=X.columns[categorical_mask])
    total_features_after_encoding = X_encoded.shape[1]

    numerical_features = X.loc[:, [not is_cat for is_cat in categorical_mask]]

    # Calculate skewness
    skewness = numerical_features.skew().to_dict()

    # Calculate correlations
    correlations = X_encoded.corrwith(pd.Series(y)).abs()
    strong_correlations_with_target = int((correlations > 0.5).sum())

    # Pairwise feature relationships
    pairwise_correlations = X_encoded.corr().abs()
    num_pairwise_relationships = int(len(pairwise_correlations) ** 2)
    strong_pairwise_correlations = int(((pairwise_correlations > 0.5).sum().sum() - len(pairwise_correlations)) // 2)

    return {
        "dataset_name": dataset.name,
        "num_classes": int(y.nunique()),
        "num_samples": int(X.shape[0]),
        "num_features": int(num_features),
        "num_categorical_features": int(num_categorical_features),
        "num_continuous_features": int(num_continuous_features),
        "class_distribution": {str(k): int(v) for k, v in class_distribution.items()},
        "total_features_after_encoding": int(total_features_after_encoding),
        "skewness": {str(k): float(v) for k, v in skewness.items()},
        "strong_correlations_with_target": strong_correlations_with_target,
        "num_pairwise_relationships": num_pairwise_relationships,
        "strong_pairwise_correlations": strong_pairwise_correlations
    }

def generate_dataset_info(data_dir):
    os.makedirs(data_dir, exist_ok=True)

    for dataset_name in DATASET_OUT_DOMAIN_MAP.keys():
        dataset_info_path = os.path.join(data_dir, dataset_name, "dataset_info.json")
        os.makedirs(os.path.join(data_dir, dataset_name), exist_ok=True)

        task_id = DATASET_OUT_DOMAIN_MAP[dataset_name]
        dataset_info = get_dataset_info(task_id)

        with open(dataset_info_path, 'w') as f:
            json.dump(dataset_info, f, indent=4)

        print(f"Dataset information saved to {dataset_info_path}")


if __name__ == "__main__":
    args = parse_args()
    generate_dataset_info(args.data_dir)
