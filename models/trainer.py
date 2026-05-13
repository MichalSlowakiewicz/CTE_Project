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
        # Calculate scale_pos_weight to handle imbalanced datasets (ecom-offers)
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        spw = num_neg / num_pos if num_pos > 0 else 1.0
        
        params = {
            "n_estimators": 300, # Increased, will be controlled by early stopping
            "max_depth": 4,
            "learning_rate": 0.1,
            "random_state": 42,
            "eval_metric": "logloss",
            "scale_pos_weight": spw,
            "early_stopping_rounds": 10 if (X_val is not None and y_val is not None) else None
        }

    model = xgb.XGBClassifier(**params)
    
    if X_val is not None and y_val is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
    else:
        model.fit(X_train, y_train)

    if verbose:
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        print(f"  [model] Train AUC: {train_auc:.4f}", end="")
        if X_val is not None and y_val is not None:
            val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            print(f"  Val AUC: {val_auc:.4f}", end="")
        print()

    return model
