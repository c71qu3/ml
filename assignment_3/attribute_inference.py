from sklearn.pipeline import Pipeline
import pandas as pd
from typing import List, Any
import numpy as np


def attribute_inference(
        model_variant: Pipeline,
        X_data: pd.DataFrame,
        feature_column: str,
        feature_values: List[Any]
    ) -> pd.DataFrame:
    """
    Performs attribute inference attack on a single feature.
    """
    mappings = []
    
    for _, row in X_data.iterrows():
        victim_row = row.copy()
        true_value = victim_row[feature_column]
        
        # Generate hypothetical records (one per possible value)
        attack_rows = []
        for value in feature_values:
            hypothetical_row = victim_row.copy()
            hypothetical_row[feature_column] = value
            # hypothetical_row['guess_value'] = value
            attack_rows.append(hypothetical_row)
        
        # Get prediction probabilities
        X_attack = pd.DataFrame(attack_rows)
        probabilities = model_variant.predict_proba(X_attack)
        confidences = np.max(probabilities, axis=1)
        
        # Select guess with highest confidence
        best_index = np.argmax(confidences)
        inferred_value = X_attack.iloc[best_index][feature_column]
        
        mappings.append({
            'true_value': true_value,
            'inferred_value': inferred_value,
            'is_correct': true_value == inferred_value
        })
    
    return pd.DataFrame(mappings)