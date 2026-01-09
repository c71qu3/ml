import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Union, Type, Any

from sklearn.preprocessing import (
    OneHotEncoder, StandardScaler, LabelEncoder)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import warnings
import datetime


def separate_features(
        data_frame: pd.DataFrame,
        target_column: str
    ) -> Tuple[pd.DataFrame, pd.Series]:
    """Return features X and target y separately."""
    X = data_frame.drop(target_column, axis=1)
    y = data_frame[target_column]
    return X, y


def encode_target(
        y_raw: pd.Series
    ) -> np.ndarray:
    """Returns encoded ndarray of target series."""
    is_object = y_raw.dtype == 'object'
    is_categorical = isinstance(y_raw, pd.CategoricalDtype)
    if is_object or is_categorical:
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
    else:
        y = y_raw.values
    return y


def calculate_baselines(
        data_frame: pd.DataFrame,
        categorical_columns: List[str]
    ) -> Dict[str, Union[int, float]]:
    """Return baseline values for attack attempts."""
    baselines = {}
    for column in categorical_columns:
        class_probs = data_frame[column].value_counts(normalize=True)
        baselines[column] = {
            'random': float((class_probs ** 2).sum()),
            'majority': float(class_probs.max()),
            'cardinality': data_frame[column].nunique()}
    return baselines


def preprocessing_pipeline(
        categorical_columns: List[str],
        numerical_columns: List[str]
    ) -> ColumnTransformer:
    """Returns pre-processing pipeline for given features."""
    return ColumnTransformer(
        transformers=[
            (
                'cat',
                OneHotEncoder(handle_unknown='ignore'),
                categorical_columns),
            (
                'num',
                StandardScaler(),
                numerical_columns)])


DELTA = {
    "model__n_estimators": (-2, 2, ),
    "model__max_depth": (-2, 2),
    "model__learning_rate": (-2, 2),
    "model__min_samples_split": (2, -2),
    "model__min_samples_leaf": (2, -2)}


def train_variants(
        model_estimator: Any,
        X_trn: pd.DataFrame,
        y_trn: np.ndarray,
        data_preprocessor: ColumnTransformer,
        model_config: Dict[str, Any],
        random_state: int=42
    ):
    """Return specified model trained with given data."""
    variants = {}
    random_seed = {'random_state': random_state}

    # Optimal model
    base_model = model_estimator(
        **random_seed,
        **model_config['base_params'])
    base_pipeline = Pipeline([
        ('preprocessor', data_preprocessor),
        ('model', base_model)])

    grid_search = GridSearchCV(
        estimator=base_pipeline,
        param_grid=model_config['grid_search_params'],
        cv=3, n_jobs=-1, verbose=0)
    grid_search.fit(X_trn, y_trn)
    optimal_params = grid_search.best_params_
    variants['Optimal'] = grid_search.best_estimator_

    # Get underfit parameters
    underfit_params = {}
    for param, optimal_value in optimal_params.items():
        params = model_config['grid_search_params'][param]
        delta = DELTA[param][0]
        if delta < 0:
            i = max(0, params.index(optimal_value) + delta)
        else:
            i = min(len(params) - 1, params.index(optimal_value) + delta)
        underfit_value = params[i]
        if underfit_value == optimal_value:
            message = f"GridSearch parameters may be too limited: `{param[7:]}`"
            warnings.warn(message, UserWarning)
        underfit_params[param[7:]] = underfit_value

    # Underfitted model
    underfit_model = model_estimator(
        **random_seed,
        **underfit_params,
        **model_config['base_params'])
    underfit_pipeline = Pipeline([
        ('preprocessor', data_preprocessor),
        ('model', underfit_model)])
    underfit_pipeline.fit(X_trn, y_trn)
    variants['Underfit'] = underfit_pipeline

    # Get overfit parameters
    overfit_params = {}
    for param, optimal_value in optimal_params.items():
        params = model_config['grid_search_params'][param]
        delta = DELTA[param][1]
        if delta > 0:
            i = min(len(params) - 1, params.index(optimal_value) + delta)
        else:
            i = max(0, params.index(optimal_value) + delta)
        overfit_value = params[i]
        if overfit_value == optimal_value:
            message = f"GridSearch parameters may be too limited: `{param[7:]}`"
            warnings.warn(message, UserWarning)
        overfit_params[param[7:]] = overfit_value

    # Overfitted model
    overfit_model = model_estimator(
        **random_seed,
        **overfit_params,
        # **model_config['overfit_params'],
        **model_config['base_params'])
    overfit_pipeline = Pipeline([
        ('preprocessor', data_preprocessor),
        ('model', overfit_model)
    ])
    overfit_pipeline.fit(X_trn, y_trn)
    variants['Overfit'] = overfit_pipeline

    optimal_params = {
        key[7:]: value
        for key, value
        in optimal_params.items()}
    optimal_params['variant'] = 'Optimal'
    underfit_params['variant'] = 'Underfit'
    overfit_params['variant'] = 'Overfit'
    parameters = [
        underfit_params,
        optimal_params,
        overfit_params]

    return variants, parameters