import glob
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def get_latest_file(
    filename: str,
    directory: str=os.path.join("data", "output")
) -> pd.DataFrame:
    """Returns the content of the most recent CSV matching the filename."""
    pattern = re.compile(rf'(\d{{8}})_.*_{re.escape(filename)}\.csv$')
    files = glob.glob(os.path.join(directory, f'*_{filename}.csv'))
    latest_file = None
    latest_date = None
    for f in files:
        match = pattern.search(os.path.basename(f))
        if match:
            file_date = match.group(1)
            if (latest_date is None) or (file_date > latest_date):
                latest_date = file_date
                latest_file = f
    if latest_file:
        return pd.read_csv(latest_file)
    return None


def merge_result_features(
        attack: pd.DataFrame,
        importance: pd.DataFrame
    ) -> pd.DataFrame:
    """Merge attack results with feature importance."""

    attack_features = attack["feature"].unique()
    importance_features = importance["feature"].unique()

    def find_base_feature(importance_feature):
        for base_feature in attack_features:
            if base_feature in importance_feature:
                return base_feature
        return None

    importance['base_feature'] = importance['feature'].apply(find_base_feature)
    importance = importance[importance['base_feature'].notna()]
    importance = importance.groupby(
        ['model', 'variant', 'base_feature'],
        as_index=False)['importance'].sum()

    merged = pd.merge(
        attack,
        importance,
        left_on=['model', 'variant', 'feature'],
        right_on=['model', 'variant', 'base_feature'])
    return merged


def correlation_attack_importance(timestamp: str) -> None:
    """Generate figurre for correlation between attack accuracy and feature importance."""
    importance = get_latest_file("feature_importance")
    attack = get_latest_file("attack_results")
    attack = attack[attack["split"] == "Test"]

    all_datasets = attack["dataset"].unique()
    merged = pd.DataFrame({
        "dataset": [], "feature": [], "model": [], "variant": [],
        "random": [], "majority": [], "accuracy": [], "importance": []})
    for dataset in all_datasets:
        merged = pd.concat(
            [merged, merge_result_features(
                attack[attack["dataset"] == dataset],
                importance[importance["dataset"] == dataset])],
            ignore_index=True)
    print(merged["dataset"].unique())

    os.makedirs("images", exist_ok=True)
    path = os.path.join("images", f"{timestamp}_importance_vs_accuracy.png")

    fig, ax = plt.subplots()
    for idx, dataset in enumerate(merged['dataset'].unique()):
        data = merged[merged['dataset'] == dataset]
        ax.scatter(
            data['importance'], data['accuracy'],
            label=dataset
        )
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Inference Attack Accuracy')
    ax.set_title('Feature Importance vs. Attack Accuracy')
    ax.legend(title='Dataset')

    plt.savefig(path)


# def correlation_attack_unique(timestamp: str) -> None:

#     attack = get_latest_file("attack_results")
#     attack = attack[attack["split"] == "Test"]

#     all_datasets = attack["dataset"].unique()
#     # merged = pd.DataFrame({
#     #     "dataset": [], "feature": [], "model": [], "variant": [],
#     #     "random": [], "majority": [], "accuracy": [], "importance": []})
#     for dataset in all_datasets:
#         all_features = attack[attack["dataset"] == dataset]["feature"].unique()
#         data = pd.read_csv(f'{dataset}.csv')
#         merged = pd.concat(
#             [merged, merge_result_features(
#                 attack[attack["dataset"] == dataset],
#                 importance[importance["dataset"] == dataset])],
#             ignore_index=True)
#     print(merged["dataset"].unique())

#     os.makedirs("images", exist_ok=True)
#     path = os.path.join("images", "importance_vs_attack_accuracy.png")

#     fig, ax = plt.subplots()
#     for idx, dataset in enumerate(merged['dataset'].unique()):
#         data = merged[merged['dataset'] == dataset]
#         ax.scatter(
#             data['importance'], data['accuracy'],
#             label=dataset
#         )
#     ax.set_xlabel('Feature Importance')
#     ax.set_ylabel('Inference Attack Accuracy')
#     ax.set_title('Feature Importance vs. Attack Accuracy')
#     ax.legend(title='Dataset')

#     plt.savefig(path)


if __name__ == "__main__":

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    correlation_attack_importance(timestamp)
    # correlation_attack_unique(timestamp)