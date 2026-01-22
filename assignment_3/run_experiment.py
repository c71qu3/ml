import os
import json
import pandas as pd
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from helper_functions import *
from attribute_inference_parallel import *

import argparse
import datetime

import time
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
DATA_DIRECTORY = os.path.join('.', 'data')

MODEL_CLASS = {
    'XGBoost': XGBClassifier,
    'Decision Tree': DecisionTreeClassifier,
    'Random Forest': RandomForestClassifier}


def run_experiment(
        model: Pipeline,
        X: pd.DataFrame,
        feature: str,
        values: List[Any]
    ) -> Dict[str, Any]:
    """Attemp an attack on a model for given data."""
    start = time.time()
    results = attribute_inference(model, X, feature, values)
    accuracy = results['is_correct'].mean()
    end = time.time()
    return accuracy, end - start


def main(
        data_config: Dict[str, Any],
        model_config: Dict[str, Any],
        data_directory: str=DATA_DIRECTORY,
        random_state: int=RANDOM_STATE
    ) -> Dict[str, pd.DataFrame]:
    """"""
    # Iterate over each dataset
    all_parameters = []
    attack_results = pd.DataFrame({
        'dataset': [], 'feature': [], 'model': [],
        'variant': [], 'split': [], 'random': [],
        'majority': [], 'accuracy': [], 'duration': []})
    tree_importances = pd.DataFrame({
        'dataset': [], 'model': [], 'variant': [],
        'feature': [], 'importance': []})
    for dataset in data_config:

        # Load dataset
        filename = data_config[dataset]['filename']
        df = pd.read_csv(os.path.join(data_directory, filename))

        # Separate features and target
        target = data_config[dataset]['target']
        X, y_raw = separate_features(df, target)
        y = encode_target(y_raw)

        # Split data into train and test
        X_trn, X_tst, y_trn, y_tst = train_test_split(
            X, y,
            test_size=0.2,
            random_state=random_state,
            stratify=y)

        # Calculate dataset  baseline
        categorical = data_config[dataset]['categorical_features']
        baselines = calculate_baselines(df, categorical)

        # Define pre-processor pipeline
        numerical = data_config[dataset]['numerical_features']
        data_preprocessor = preprocessing_pipeline(categorical, numerical)

        # Train each available model variants
        trained_variants = {}
        for model_name in model_config:
            trained_variants[model_name], parameters = train_variants(
                MODEL_CLASS[model_name],
                X_trn, y_trn,
                data_preprocessor,
                model_config[model_name],
                random_state)
            for row in parameters:
                row['model'] = model_name
                row['dataset'] = dataset
            all_parameters += parameters

        # Save decision tree feature importance
        for m in trained_variants:
            for v, p in trained_variants[m].items():
                model = p.named_steps['model']
                features = p.named_steps['preprocessor'].get_feature_names_out()
                importances = model.feature_importances_
                new_rows = pd.DataFrame({
                    "dataset": [dataset] * len(features),
                    "model": [m] * len(features),
                    "variant": [v] * len(features),
                    "feature": features,
                    "importance": importances})
                tree_importances = pd.concat([tree_importances, new_rows], ignore_index=True)

        # Iterate over each categorical feature
        results_rows = []
        for feature in categorical:
            feature_values = df[feature].unique()
            feature_baseline = baselines[feature]

            # Iterate over each model variant
            for model_name in model_config:
                model_variants = trained_variants[model_name]
                for variant_name, model_variant in model_variants.items():

                    for X, split in zip([X_trn, X_tst], ['Train', 'Test']):
                        accuracy, duration = run_experiment(
                            model_variant, X,
                            feature, feature_values)
                        results_rows.append({
                            'dataset': dataset,
                            'feature': feature,
                            'model': model_name,
                            'variant': variant_name,
                            'split': split,
                            'random': baselines[feature]['random'],
                            'majority': baselines[feature]['majority'],
                            'accuracy': accuracy,
                            'duration': duration})

        attack_results = pd.concat(
            [attack_results, pd.DataFrame(results_rows)],
            ignore_index=True)
    return attack_results, all_parameters, tree_importances


if __name__ == "__main__":

    model_config = {
        "XGBoost": {
            "base_params": {
            "eval_metric": "logloss"
            },
            "grid_search_params": {
            "model__n_estimators": [4, 8, 12, 16, 24, 32, 64, 128, 256],
            "model__learning_rate": [0.001, 0.01, 0.1, 0.2, 0.4, 0.8],
            "model__max_depth": [1, 2, 3, 4, 6, 8, 12, 16]
            }
        },
        "Decision Tree": {
            "base_params": {},
            "grid_search_params": {
            "model__max_depth": [1, 2, 3, 4, 6, 8, 12, 16],
            "model__min_samples_split": [2, 4, 6, 8, 10, 12, 14],
            "model__min_samples_leaf": [1, 2, 4, 6, 8, 12, 16, 22]
            }
        },
        "Random Forest": {
            "base_params": {
            "n_jobs": 1
            },
            "grid_search_params": {
            "model__n_estimators": [10, 20, 30, 60, 90, 120, 160, 200, 240, 280, 320],
            "model__min_samples_split": [2, 4, 6, 8, 10, 12, 14],
            "model__max_depth": [1, 2, 3, 4, 6, 8, 12, 16]
            }
        }
    }

    description = "Run ML attribute inference experiment."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '-d', '--data-config',
        type=str, required=False,
        default=os.path.join(DATA_DIRECTORY, "all_datasets.json"),
        help="Filepath to JSON file settings for experiment datasets.")

    # parser.add_argument(
    #     '-m', '--model-config',
    #     type=str, required=True,
    #     help="Filepath to model_config JSON file.")

    parser.add_argument(
        '--data-dir',
        type=str, required=False,
        default=DATA_DIRECTORY,
        help="Data directory for experiment datasets.")

    parser.add_argument(
        '--random-state',
        type=int, required=False,
        default=RANDOM_STATE,
        help="Random state for reproducibility.")

    args = parser.parse_args()

    with open(args.data_config, 'r') as file:
        data_config = json.load(file)

    # with open(args.model_config, 'r') as file:
    #     model_config = json.load(file)

    results, parameters, importances = main(
        data_config=data_config,
        model_config=model_config,
        data_directory=args.data_dir,
        random_state=args.random_state)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    filename = f"{timestamp}_attack_results.csv"
    results.to_csv(
        os.path.join(DATA_DIRECTORY, "output", filename),
        index=False)

    parameters = pd.DataFrame(parameters)
    filename = f"{timestamp}_experiment_parameters.csv"
    parameters.to_csv(
        os.path.join(DATA_DIRECTORY, "output", filename),
        index=False)

    filename = f"{timestamp}_feature_importance.csv"
    importances.to_csv(
        os.path.join(DATA_DIRECTORY, "output", filename),
        index=False)