import datetime as dt

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    precision_recall_fscore_support, precision_recall_curve, average_precision_score
)

from xgboost import XGBClassifier


def train_test_split_print(df: pd.DataFrame, lbl_column: str, test_pct: float = 0.3, seed: int = 42) -> tuple:
    df_train, df_test = train_test_split(
            df,
            test_size=test_pct,
            stratify=df[lbl_column],
            random_state=seed
    )    

    print(f"Train rows: {df_train.shape[0]}")
    print(f"Test rows: {df_test.shape[0]}")

    return df_train, df_test


def get_X_y(df: pd.DataFrame, lbl_col: str) -> tuple:
    X = df[[c for c in df.columns if c != lbl_col]]
    y = df[lbl_col]

    return X, y


def print_prfs(prfs: list, label: str) -> None:
    display(pd.DataFrame({
        f"{label} Precision": [prfs[0]], 
        f"{label} Recall": [prfs[1]], 
        f"{label} F1": [prfs[2]], 
    }))


def display_train_test_confusion_matrix(y_train: np.array, y_test: np.array, train_y_pred: np.array, test_y_pred: np.array) -> None:
    cm_train = confusion_matrix(y_train, train_y_pred)
    cm_test = confusion_matrix(y_test, test_y_pred)
    
    fig, ax = plt.subplots(1,2, figsize=(12, 6))
    
    cm_train_display = ConfusionMatrixDisplay(cm_train)
    cm_train_display.plot(ax=ax[0])
    
    cm_test_display = ConfusionMatrixDisplay(cm_test)
    cm_test_display.plot(ax=ax[1])
    
    ax[0].title.set_text("Train Confusion Matrix")
    ax[1].title.set_text("Test Confusion Matrix")
    
    plt.tight_layout()


def prepare_and_run_cv_baseline(df: pd.DataFrame, 
                                model_type: str,
                                lbl_column: str,
                                cv_validation_pct: float = 0.3,
                                cv_seeds: list = [1, 7, 42, 100, 120],
                                log_reg_params: list = None,
                                xgb_classifier_params: list = None,
                                svm_classifier_params: list = None,
                               ):
    fold_validation_results = []
    experiment_details = []
    for idx, seed in enumerate(cv_seeds):
        start_fold_time = dt.datetime.now()
        print(f"{start_fold_time.strftime('%Y%m%d %H:%M:%S')} - Training for {idx+1}th fold out of {len(cv_seeds)}.")
        # Split dataset into train and test
        df_train, df_validation = train_test_split_print(df=df,
                                                   lbl_column=lbl_column,
                                                   test_pct=cv_validation_pct,
                                                   seed=seed
                                                  )

        # We impute the price with the mean of the training dataset split to avoid leakage
        train_price_mean_fill_na = df_train["WINE_INPUT_PRICE"].mean()
        df_train["WINE_INPUT_PRICE"] = df_train["WINE_INPUT_PRICE"].fillna(train_price_mean_fill_na)
        df_validation["WINE_INPUT_PRICE"] = df_validation["WINE_INPUT_PRICE"].fillna(train_price_mean_fill_na)
    
        # Separate the features from the label
        X_train, y_train = get_X_y(df=df_train, lbl_col=lbl_column)
        X_validation, y_validation = get_X_y(df=df_validation, lbl_col=lbl_column)
    
        # Encode the label as numeric, by fitting on the training split and applying over the testing split
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_validation = le.transform(y_validation)
    
        # Specify the columns which are categorical
        cat_cols = ["WINE_INPUT_PROVINCE", "WINE_INPUT_COUNTRY_CODE"]
        X_train_num = X_train[[c for c in X_train.columns if c not in cat_cols]]
        X_validation_num = X_validation[[c for c in X_validation.columns if c not in cat_cols]]
    
        # Encode the categoricals, by fitting on the training split and then applying over the testing split
        # This way we make sure that for each fold there is no leakage for categories which might exist in test but not in train for that loop.
        one_hot_enc = OneHotEncoder(handle_unknown='ignore', 
                                    sparse_output=False, 
                                    drop="first")
        X_train_enc = one_hot_enc.fit_transform(X_train[cat_cols])
        X_validation_enc = one_hot_enc.transform(X_validation[cat_cols])
    
        # We rejoin the numeric and categorical encoded columns
        X_train = np.hstack((X_train_num, X_train_enc))
        X_validation = np.hstack((X_validation_num, X_validation_enc))
    
        # Scale data to make it more efficient even for models which dont need it, by fitting on the training split and applying over the testing split
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        X_train = scaler.transform(X_train)
        X_validation = scaler.transform(X_validation)

        models = []
        model_params = []
        if model_type.upper() in ["LR", "LOGISTICREGRESSION", "LOGISTIC_REGRESSION", "LOGREG", "LOG_REG"]:
            if log_reg_params:
                for model_params in log_reg_params:
                    log_reg = LogisticRegression(random_state=seed, 
                                                 **model_params
                    )
                    models.append(log_reg)
                    model_params = log_reg_params
            else:
                log_reg = LogisticRegression(random_state=seed)
                models.append(log_reg)
        elif model_type.upper() in ["XGB", "XGBC", "XGBOOST", "XGB_CLASSIFIER", "XGBCLASSIER", "XGBOOSTCLASSIFIER", "XGBOOST_CLASSIFIER"]:
            if xgb_classifier_params:
                for model_params in xgb_classifier_params:
                    xgbc = XGBClassifier(random_state=seed, 
                                         **model_params
                    )
                    models.append(xgbc)
                    model_params = xgb_classifier_params
            else:
                xgbc = XGBClassifier(random_state=seed)
                models.append(xgbc)
        elif model_type.upper() in ["SVM", "SVMC", "SVM_C", "SVMCLASSIFIER", "SVM_CLASSIFIER"]:
            if svm_classifier_params:
                for model_params in svm_classifier_params:
                    svmc = SVC(random_state=seed, 
                                   kernel="linear", 
                                   cache_size=1000, 
                                   verbose=True, 
                                   **model_params
                    )
                    models.append(svmc)
                    model_params = svm_classifier_params
            else:
                svmc = SVC(random_state=seed, 
                                   kernel="linear", 
                                   cache_size=1000, 
                                   verbose=True
                )
                models.append(svmc)
        else:
            raise Exception(f"Invalid model type: {model_type.upper()}")

        models_folds_results = []
        for idx_m, model in enumerate(models):
            start_time = dt.datetime.now()
            print(f"  {start_time.strftime('%Y/%m/%d %H:%M:%S')} - Training {idx_m+1}th model out of {len(models)}")
            # Fit model
            model.fit(X_train, y_train)

            # Make predictions
            train_y_pred = model.predict(X_train)
            validation_y_pred = model.predict(X_validation)

            # Calculate Precision, Recall and F1 score
            train_prfs = precision_recall_fscore_support(y_train, train_y_pred, average=None)
            validation_prfs = precision_recall_fscore_support(y_validation, validation_y_pred, average=None)

            train_prfs_macro = precision_recall_fscore_support(y_train, train_y_pred, average="macro")
            validation_prfs_macro = precision_recall_fscore_support(y_validation, validation_y_pred, average="macro")
        
            fold_train_metrics_df = pd.DataFrame(
                {
                    "Class": [c for c in range(train_prfs[0].shape[0])],
                    "Precision": train_prfs[0],
                    "Recall": train_prfs[1],
                    "F1": train_prfs[2],
                }
            )
            fold_train_metrics_df["Fold"] = idx
            #train_prfs_arr.append(fold_train_metrics_df)
            
            fold_validation_metrics_df = pd.DataFrame(
                {
                    "class": [c for c in range(validation_prfs[0].shape[0])],
                    "Precision": validation_prfs[0],
                    "Recall": validation_prfs[1],
                    "F1": validation_prfs[2],
                }
            )
            fold_validation_metrics_df["Fold"] = idx
            #validation_prfs_arr.append(fold_validation_metrics_df)

            experiment_details.append({
                "CONFIG": {
                    "MODEL": model,
                    "LABEL_ENCODER": le,
                    "CATEGORICAL_ENCODER": one_hot_enc,
                    "NUMERIC_SCALER": scaler,
                    "PARAMS": model_params[idx_m]
                },
                "DATA": {
                    "X_TRAIN": X_train,
                    "X_VALIDATION": X_validation,
                    "Y_TRAIN": y_train,
                    "Y_VALIDATION": y_validation,
                },
                "RESULTS": {
                    "PRED_TRAIN": train_y_pred,
                    "PRED_VALIDATION": validation_y_pred,
                    "VALIDATION_PRECISION": validation_prfs_macro[0],
                    "VALIDATION_RECALL": validation_prfs_macro[1],
                    "VALIDATION_F1": validation_prfs_macro[2],
                }
            })
            end_time = dt.datetime.now()
            print(f"  {end_time.strftime('%Y/%m/%d %H:%M:%S')} - Duration: {end_time-start_time}.")
            models_folds_results.append({str(model_params[idx_m]): validation_prfs_macro})

        fold_validation_results.append(models_folds_results)
    
    end_fold_time = dt.datetime.now()
    print(f"{end_fold_time.strftime('%Y%m%d %H:%M:%S')} - Duration: {end_fold_time-start_fold_time}.")
    return experiment_details, fold_validation_results


def prepare_and_run_cv_feat_eng(df: pd.DataFrame, 
                                model_type: str,
                                lbl_column: str,
                                cv_validation_pct: float = 0.3,
                                cv_seeds: list = [1, 7, 42, 100, 120],
                                log_reg_params: list = None,
                                xgb_classifier_params: list = None,
                                svm_classifier_params: list = None,
                               ):
    fold_validation_results = []
    experiment_details = []

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=1000, 
        min_df=2, 
        max_df=0.9,
    )
    
    for idx, seed in enumerate(cv_seeds):
        start_fold_time = dt.datetime.now()
        print(f"{start_fold_time.strftime('%Y%m%d %H:%M:%S')} - Training for {idx+1}th fold out of {len(cv_seeds)}.")
        # Split dataset into train and test
        df_train, df_validation = train_test_split_print(df=df,
                                                   lbl_column=lbl_column,
                                                   test_pct=cv_validation_pct,
                                                   seed=seed
                                                  )

        # Extract TF-IDF vectors from training and apply to validation
        X_train_tfidf = vectorizer.fit_transform(df_train["WINE_INPUT_DESCRIPTION"])
        X_validation_tfidf = vectorizer.transform(df_validation["WINE_INPUT_DESCRIPTION"])

        df_train = df_train.drop("WINE_INPUT_DESCRIPTION", axis=1)
        df_validation = df_validation.drop("WINE_INPUT_DESCRIPTION", axis=1)

        # We impute the price with the mean of the training dataset split to avoid leakage
        train_price_mean_fill_na = df_train["WINE_INPUT_PRICE"].mean()
        df_train["WINE_INPUT_PRICE"] = df_train["WINE_INPUT_PRICE"].fillna(train_price_mean_fill_na)
        df_validation["WINE_INPUT_PRICE"] = df_validation["WINE_INPUT_PRICE"].fillna(train_price_mean_fill_na)
    
        # Separate the features from the label
        X_train, y_train = get_X_y(df=df_train, lbl_col=lbl_column)
        X_validation, y_validation = get_X_y(df=df_validation, lbl_col=lbl_column)
    
        # Encode the label as numeric, by fitting on the training split and applying over the testing split
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_validation = le.transform(y_validation)
    
        # Specify the columns which are categorical
        cat_cols = ["WINE_INPUT_PROVINCE", "WINE_INPUT_COUNTRY_CODE"]
        X_train_num = X_train[[c for c in X_train.columns if c not in cat_cols]]
        X_validation_num = X_validation[[c for c in X_validation.columns if c not in cat_cols]]
    
        # Encode the categoricals, by fitting on the training split and then applying over the testing split
        # This way we make sure that for each fold there is no leakage for categories which might exist in test but not in train for that loop.
        one_hot_enc = OneHotEncoder(handle_unknown='ignore', 
                                    sparse_output=False, 
                                    drop="first")
        X_train_enc = one_hot_enc.fit_transform(X_train[cat_cols])
        X_validation_enc = one_hot_enc.transform(X_validation[cat_cols])
    
        # We rejoin the numeric and categorical encoded columns
        X_train = np.hstack((X_train_num, X_train_enc, X_train_tfidf.toarray()))
        X_validation = np.hstack((X_validation_num, X_validation_enc, X_validation_tfidf.toarray()))
    
        # Scale data to make it more efficient even for models which dont need it, by fitting on the training split and applying over the testing split
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        X_train = scaler.transform(X_train)
        X_validation = scaler.transform(X_validation)

        models = []
        model_params = []
        if model_type.upper() in ["LR", "LOGISTICREGRESSION", "LOGISTIC_REGRESSION", "LOGREG", "LOG_REG"]:
            if log_reg_params:
                for model_params in log_reg_params:
                    log_reg = LogisticRegression(random_state=seed, 
                                                 **model_params
                    )
                    models.append(log_reg)
                    model_params = log_reg_params
            else:
                log_reg = LogisticRegression(random_state=seed)
                models.append(log_reg)
        elif model_type.upper() in ["XGB", "XGBC", "XGBOOST", "XGB_CLASSIFIER", "XGBCLASSIER", "XGBOOSTCLASSIFIER", "XGBOOST_CLASSIFIER"]:
            if xgb_classifier_params:
                for model_params in xgb_classifier_params:
                    xgbc = XGBClassifier(random_state=seed, 
                                         **model_params
                    )
                    models.append(xgbc)
                    model_params = xgb_classifier_params
            else:
                xgbc = XGBClassifier(random_state=seed)
                models.append(xgbc)
        elif model_type.upper() in ["SVC", "SVM", "SVMC", "SVM_C", "SVMCLASSIFIER", "SVM_CLASSIFIER"]:
            if svm_classifier_params:
                for model_params in svm_classifier_params:
                    svmc = SVC(random_state=seed, 
                                   #kernel="linear", 
                                   cache_size=1000, 
                                   verbose=True, 
                                   **model_params
                    )
                    models.append(svmc)
                    model_params = svm_classifier_params
            else:
                svmc = SVC(random_state=seed, 
                                   kernel="linear", 
                                   cache_size=1000, 
                                   verbose=True
                )
                models.append(svmc)
        else:
            raise Exception(f"Invalid model type: {model_type.upper()}")

        models_folds_results = []
        for idx_m, model in enumerate(models):
            start_time = dt.datetime.now()
            print(f"  {start_time.strftime('%Y/%m/%d %H:%M:%S')} - Training {idx_m+1}th model out of {len(models)}")
            # Fit model
            model.fit(X_train, y_train)

            # Make predictions
            train_y_pred = model.predict(X_train)
            validation_y_pred = model.predict(X_validation)

            # Calculate Precision, Recall and F1 score
            train_prfs = precision_recall_fscore_support(y_train, train_y_pred, average=None)
            validation_prfs = precision_recall_fscore_support(y_validation, validation_y_pred, average=None)

            train_prfs_macro = precision_recall_fscore_support(y_train, train_y_pred, average="macro")
            validation_prfs_macro = precision_recall_fscore_support(y_validation, validation_y_pred, average="macro")
        
            fold_train_metrics_df = pd.DataFrame(
                {
                    "Class": [c for c in range(train_prfs[0].shape[0])],
                    "Precision": train_prfs[0],
                    "Recall": train_prfs[1],
                    "F1": train_prfs[2],
                }
            )
            fold_train_metrics_df["Fold"] = idx
            #train_prfs_arr.append(fold_train_metrics_df)
            
            fold_validation_metrics_df = pd.DataFrame(
                {
                    "class": [c for c in range(validation_prfs[0].shape[0])],
                    "Precision": validation_prfs[0],
                    "Recall": validation_prfs[1],
                    "F1": validation_prfs[2],
                }
            )
            fold_validation_metrics_df["Fold"] = idx
            #validation_prfs_arr.append(fold_validation_metrics_df)

            experiment_details.append({
                "CONFIG": {
                    "MODEL": model,
                    "LABEL_ENCODER": le,
                    "CATEGORICAL_ENCODER": one_hot_enc,
                    "NUMERIC_SCALER": scaler,
                    "PARAMS": model_params[idx_m]
                },
                "DATA": {
                    "X_TRAIN": X_train,
                    "X_VALIDATION": X_validation,
                    "Y_TRAIN": y_train,
                    "Y_VALIDATION": y_validation,
                },
                "RESULTS": {
                    "PRED_TRAIN": train_y_pred,
                    "PRED_VALIDATION": validation_y_pred,
                    "VALIDATION_PRECISION": validation_prfs_macro[0],
                    "VALIDATION_RECALL": validation_prfs_macro[1],
                    "VALIDATION_F1": validation_prfs_macro[2],
                }
            })
            end_time = dt.datetime.now()
            print(f"  {end_time.strftime('%Y/%m/%d %H:%M:%S')} - Duration: {end_time-start_time}.")
            models_folds_results.append({str(model_params[idx_m]): validation_prfs_macro})

        fold_validation_results.append(models_folds_results)
    
    end_fold_time = dt.datetime.now()
    print(f"{end_fold_time.strftime('%Y%m%d %H:%M:%S')} - Duration: {end_fold_time-start_fold_time}.")
    return experiment_details, fold_validation_results


def plot_cv_fold_boxplot(fold_validation_results):
    transformed = {}

    for i in range(len(fold_validation_results[0])):
        key = list(fold_validation_results[0][i].keys())[0]
        vals = [list(d[i].values())[0] for d in fold_validation_results] 
        transformed[key] = vals

    
    fig, ax = plt.subplots(1, len(transformed.keys()), figsize=(25, 6))
    
    for idx, t in enumerate(transformed.items()):
        params, fold_results = t
        a = np.array(fold_results)
        df_metrics = pd.DataFrame(a).loc[:, [0,1,2]]
        df_metrics.columns = ["Precision", "Recall", "F1"]
        sns.boxplot(df_metrics, ax=ax[idx])
        ax[idx].title.set_text(f"CV results for params \n {params}")
        ax[idx].set_ylim(0.0, 1.0)
    
    plt.tight_layout()