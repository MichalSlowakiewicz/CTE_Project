import time
import joblib
import pandas as pd
from pathlib import Path
import warnings
import sys
import optuna

sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import load_dataset
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, log_loss

def objective(trial, X_train, y_train, X_val, y_val, num_neg, num_pos):
    spw = num_neg / num_pos if num_pos > 0 else 1.0
    
    params = {
        "iterations": trial.suggest_int("iterations", 100, 800),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 100.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "bootstrap_type": "Bernoulli",
        "random_seed": 42,
        "eval_metric": "Logloss",
        "scale_pos_weight": spw,
        "task_type": "GPU",
        "verbose": False
    }
    
    try:
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
    except Exception:
        params["task_type"] = "CPU"
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
        
    preds = model.predict_proba(X_val)[:, 1]
    loss = log_loss(y_val, preds)
    return loss

def train_and_optimize():
    print("=== CatBoost Training & Optuna Optimization ===")
    
    cache_dir = Path(__file__).parent.parent / 'data' / 'cache_full'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Loading FULL dataset...")
    data = load_dataset(data_dir=Path(__file__).parent.parent / 'data' / 'ecom-offers')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    
    print(f"\n2. Starting Optuna optimization on {len(X_train)} samples...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    
    n_trials = 50
    print(f"   Running {n_trials} trials for CatBoost...")
    
    t0 = time.time()
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, num_neg, num_pos), n_trials=n_trials, show_progress_bar=True)
    t1 = time.time()
    
    print(f"\n   Optimization finished in {t1-t0:.1f} seconds!")
    print("   Best parameters:")
    best_params = study.best_params
    for k, v in best_params.items():
        print(f"      {k}: {v}")
        
    print("\n3. Training final CatBoost model with best parameters...")
    spw = num_neg / num_pos if num_pos > 0 else 1.0
    final_params = {
        **best_params,
        "bootstrap_type": "Bernoulli",
        "random_seed": 42,
        "eval_metric": "Logloss",
        "scale_pos_weight": spw,
        "verbose": False
    }
    
    try:
        final_model = CatBoostClassifier(**final_params, task_type="GPU")
        final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
        print("   Final model trained successfully on GPU.")
    except Exception:
        final_model = CatBoostClassifier(**final_params, task_type="CPU")
        final_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
        print("   Final model trained successfully on CPU.")
        
    train_auc = roc_auc_score(y_train, final_model.predict_proba(X_train)[:, 1])
    val_auc = roc_auc_score(y_val, final_model.predict_proba(X_val)[:, 1])
    test_auc = roc_auc_score(data['y_test'], final_model.predict_proba(data['X_test'])[:, 1])
    
    print(f"   [model] Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")

    model_path = cache_dir / 'catboost_model.pkl'
    joblib.dump(final_model, model_path)
    print(f"\n=== Model saved to {model_path} ===")

if __name__ == "__main__":
    train_and_optimize()
