"""
models/trainer.py - XGBoost model training.

Keeps training logic in one place so experiments can just call train_model()
and get back a fitted model along with some basic performance numbers.
"""

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score


def train_model(X_train, y_train, X_val=None, y_val=None, params=None,
                verbose=True):
    """
    Train an XGBoost classifier and report AUC on train and val sets.

    Args:
        X_train:  pd.DataFrame with training features.
        y_train:  Array-like with binary training labels.
        X_val:    Optional validation features (for reporting only).
        y_val:    Optional validation labels (for reporting only).
        params:   Dict of XGBoost hyperparameters. Uses defaults if None.
        verbose:  Print AUC scores after training.

    Returns:
        Fitted xgb.XGBClassifier.
    """
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": 42,
            "eval_metric": "logloss",
        }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    if verbose:
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        print(f"  [model] Train AUC: {train_auc:.4f}", end="")
        if X_val is not None and y_val is not None:
            val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            print(f"  Val AUC: {val_auc:.4f}", end="")
        print()

    return model
