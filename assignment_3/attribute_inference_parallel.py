from sklearn.pipeline import Pipeline
import pandas as pd
from typing import List, Any
import numpy as np
from sklearn.utils.parallel import Parallel, delayed
# import warnings


# warnings.filterwarnings('ignore', category=warnings.UserWarning)


def _attack_row(row, model_variant, feature_column, feature_values):
    true_value = row[feature_column]
    attack_rows = []
    for value in feature_values:
        hypothetical_row = row.copy()
        hypothetical_row[feature_column] = value
        attack_rows.append(hypothetical_row)
    X_attack = pd.DataFrame(attack_rows)
    probabilities = model_variant.predict_proba(X_attack)
    confidences = np.max(probabilities, axis=1)
    best_index = np.argmax(confidences)
    inferred_value = X_attack.iloc[best_index][feature_column]
    return {
        # 'true_value': true_value,
        # 'inferred_value': inferred_value,
        'is_correct': true_value == inferred_value
    }

def attribute_inference(
        model_variant: Pipeline,
        X_data: pd.DataFrame,
        feature_column: str,
        feature_values: List[Any]
    ) -> pd.DataFrame:
    """
    Performs attribute inference attack on a single feature.
    """
    results = Parallel(n_jobs=4)(
        delayed(_attack_row)(row, model_variant, feature_column, feature_values)
        for _, row in X_data.iterrows()
    )
    return pd.DataFrame(results)