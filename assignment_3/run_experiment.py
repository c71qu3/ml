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
    'Random Forest': RandomForestClassifier,
    'Multi-Layer Perceptron': MLPClassifier}


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
    all_results = {}
    all_parameters = []
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

        # Iterate over each categorical feature
        attack_results = []
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
                        attack_results.append({
                            'Dataset': dataset,
                            'Feature': feature,
                            'Model': model_name,
                            'Variant': variant_name,
                            'Split': split,
                            'Random': baselines[feature]['random'],
                            'Majority': baselines[feature]['majority'],
                            'Accuracy': accuracy,
                            'Duration': duration})

        all_results[dataset] = pd.DataFrame(attack_results)
    return all_results, all_parameters


if __name__ == "__main__":

    description = "Run ML attribute inference experiment."
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument(
        '-d', '--data-config',
        type=str, required=True,
        help="Filepath to data_config JSON file.")

    parser.add_argument(
        '-m', '--model-config',
        type=str, required=True,
        help="Filepath to model_config JSON file.")

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

    with open(args.model_config, 'r') as file:
        model_config = json.load(file)

    results, parameters = main(
        data_config=data_config,
        model_config=model_config,
        data_directory=args.data_dir,
        random_state=args.random_state)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    for dataset, df in results.items():
        filename = f"{timestamp}_{dataset}_attack_results.csv"
        df.to_csv(filename, index=False)

    parameters = pd.DataFrame(parameters)
    filename = f"{timestamp}_experiment_parameters.csv"
    parameters.to_csv(filename, index=False)