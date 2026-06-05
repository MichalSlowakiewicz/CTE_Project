import time
import joblib
import pandas as pd
from pathlib import Path
import warnings
import sys
import optuna

sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import load_dataset
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss

def objective(trial, X_train, y_train, X_val, y_val, num_neg, num_pos):
    spw = num_neg / num_pos if num_pos > 0 else 1.0
    
    # Przestrzeń hiperparametrów dla LightGBM (znanego z doskonałej odporności na przeuczenie)
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "num_leaves": trial.suggest_int("num_leaves", 10, 100),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": spw,
        "verbose": -1
    }
    
    model = lgb.LGBMClassifier(**params)
    # LightGBM ma wbudowane wsparcie dla eval_set i early stopping w metodzie fit (w nowszych wersjach używamy callbacks)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
        
    preds = model.predict_proba(X_val)[:, 1]
    loss = log_loss(y_val, preds)
    return loss

def train_and_optimize():
    print("=== LightGBM Training & Optuna Optimization ===")
    
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
    print(f"   Running {n_trials} trials for LightGBM...")
    
    t0 = time.time()
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, num_neg, num_pos), n_trials=n_trials, show_progress_bar=True)
    t1 = time.time()
    
    print(f"\n   Optimization finished in {t1-t0:.1f} seconds!")
    print("   Best parameters:")
    best_params = study.best_params
    for k, v in best_params.items():
        print(f"      {k}: {v}")
        
    print("\n3. Training final LightGBM model with best parameters...")
    spw = num_neg / num_pos if num_pos > 0 else 1.0
    final_params = {
        **best_params,
        "n_jobs": -1,
        "random_state": 42,
        "scale_pos_weight": spw,
        "verbose": -1
    }
    
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    print("   Final model trained successfully.")
        
    train_auc = roc_auc_score(y_train, final_model.predict_proba(X_train)[:, 1])
    val_auc = roc_auc_score(y_val, final_model.predict_proba(X_val)[:, 1])
    test_auc = roc_auc_score(data['y_test'], final_model.predict_proba(data['X_test'])[:, 1])
    
    print(f"   [model] Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")

    model_path = cache_dir / 'lgbm_model.pkl'
    joblib.dump(final_model, model_path)
    print(f"\n=== Model saved to {model_path} ===")

if __name__ == "__main__":
    train_and_optimize()
