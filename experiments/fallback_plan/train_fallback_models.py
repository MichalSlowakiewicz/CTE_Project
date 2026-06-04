import time
import joblib
import pandas as pd
from pathlib import Path
import warnings
import sys
import optuna
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss
from interpret.glassbox import ExplainableBoostingClassifier

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.data_loader import load_dataset

def optimize_ridge(trial, X_train, y_train, X_val, y_val):
    params = {
        "alpha": trial.suggest_float("alpha", 1e-4, 10.0, log=True),
        "class_weight": "balanced",
        "random_state": 42
    }
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeClassifier(**params))
    ])
    model.fit(X_train, y_train)
    # Use decision_function as proxy for loss optimization (higher is better for AUC)
    auc = roc_auc_score(y_val, model.decision_function(X_val))
    return 1.0 - auc

def optimize_svc(trial, X_train, y_train, X_val, y_val):
    params = {
        "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
        "class_weight": "balanced",
        "random_state": 42,
        "dual": False
    }
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', LinearSVC(**params))
    ])
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_val, model.decision_function(X_val))
    return 1.0 - auc

def optimize_ebm(trial, X_train, y_train, X_val, y_val):
    params = {
        "max_bins": trial.suggest_categorical("max_bins", [128, 256, 512]),
        "max_interaction_bins": trial.suggest_categorical("max_interaction_bins", [16, 32, 64]),
        "interactions": trial.suggest_int("interactions", 0, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "random_state": 42,
        "n_jobs": -1
    }
    model = ExplainableBoostingClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_val)[:, 1]
    return log_loss(y_val, preds)

def train_and_save_fallback_models():
    print("=== Training Missing Models with Optuna ===")
    
    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Loading FULL dataset...")
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_trials = 20
    
    # ---------------------------------------------------------
    # 1. Ridge Classifier
    # ---------------------------------------------------------
    print(f"\n2. Optimizing Ridge Classifier ({n_trials} trials)...")
    study_ridge = optuna.create_study(direction="minimize")
    study_ridge.optimize(lambda t: optimize_ridge(t, X_train, y_train, X_val, y_val), n_trials=n_trials)
    print(f"   Best Ridge params: {study_ridge.best_params}")
    
    final_ridge = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', RidgeClassifier(**study_ridge.best_params, class_weight='balanced', random_state=42))
    ])
    final_ridge.fit(X_train, y_train)
    print(f"   [Ridge] Val AUC: {roc_auc_score(y_val, final_ridge.decision_function(X_val)):.4f}")
    joblib.dump(final_ridge, cache_dir / 'ridge_model.pkl')
    
    # ---------------------------------------------------------
    # 2. Linear SVC
    # ---------------------------------------------------------
    print(f"\n3. Optimizing Linear SVC ({n_trials} trials)...")
    study_svc = optuna.create_study(direction="minimize")
    study_svc.optimize(lambda t: optimize_svc(t, X_train, y_train, X_val, y_val), n_trials=n_trials)
    print(f"   Best SVC params: {study_svc.best_params}")
    
    final_svc = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', LinearSVC(**study_svc.best_params, class_weight='balanced', random_state=42, dual=False))
    ])
    final_svc.fit(X_train, y_train)
    print(f"   [LinearSVC] Val AUC: {roc_auc_score(y_val, final_svc.decision_function(X_val)):.4f}")
    joblib.dump(final_svc, cache_dir / 'svc_model.pkl')
    
    # ---------------------------------------------------------
    # 3. Explainable Boosting Machine (EBM)
    # ---------------------------------------------------------
    # EBM is quite slow to train, so we limit Optuna trials to 10
    ebm_trials = 10
    print(f"\n4. Optimizing EBM ({ebm_trials} trials)...")
    study_ebm = optuna.create_study(direction="minimize")
    study_ebm.optimize(lambda t: optimize_ebm(t, X_train, y_train, X_val, y_val), n_trials=ebm_trials)
    print(f"   Best EBM params: {study_ebm.best_params}")
    
    final_ebm = ExplainableBoostingClassifier(**study_ebm.best_params, random_state=42, n_jobs=-1)
    final_ebm.fit(X_train, y_train)
    print(f"   [EBM] Val AUC: {roc_auc_score(y_val, final_ebm.predict_proba(X_val)[:, 1]):.4f}")
    joblib.dump(final_ebm, cache_dir / 'ebm_model.pkl')
    
    print("\n=== All fallback models optimized and saved to cache! ===")

if __name__ == "__main__":
    train_and_save_fallback_models()
