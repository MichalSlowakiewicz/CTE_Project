import time
import joblib
import pandas as pd
from pathlib import Path
import warnings
import sys
import optuna

sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        "C": trial.suggest_float("C", 1e-4, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
        "penalty": "elasticnet",
        "solver": "saga",
        "max_iter": 200, # saga może potrzebować więcej iteracji, ale chcemy żeby działał w rozsądnym czasie
        "random_state": 42,
        "class_weight": "balanced",
        "n_jobs": -1
    }
    
    # Logistic Regression zawsze wymaga przeskalowanych danych!
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(**params))
    ])
    
    # Ignorujemy ostrzeżenia o braku konwergencji by nie psuć logów Optuny
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
        
    preds = model.predict_proba(X_val)[:, 1]
    loss = log_loss(y_val, preds)
    return loss

def train_and_optimize():
    print("=== Logistic Regression (ElasticNet) & Optuna Optimization ===")
    
    cache_dir = Path(__file__).parent.parent / 'data' / 'cache_full'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Loading FULL dataset...")
    data = load_dataset(data_dir=Path(__file__).parent.parent / 'data' / 'ecom-offers')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    print(f"\n2. Starting Optuna optimization on {len(X_train)} samples...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    
    # Ponieważ SAGA solver jest wolny dla 100k próbek, robimy tylko 20 iteracji
    n_trials = 20
    print(f"   Running {n_trials} trials for Logistic Regression...")
    
    t0 = time.time()
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials, show_progress_bar=True)
    t1 = time.time()
    
    print(f"\n   Optimization finished in {t1-t0:.1f} seconds!")
    print("   Best parameters:")
    best_params = study.best_params
    for k, v in best_params.items():
        print(f"      {k}: {v}")
        
    print("\n3. Training final Logistic Regression model with best parameters...")
    final_params = {
        **best_params,
        "penalty": "elasticnet",
        "solver": "saga",
        "max_iter": 500, # Dla finałowego modelu dajemy więcej iteracji na konwergencję
        "random_state": 42,
        "class_weight": "balanced",
        "n_jobs": -1
    }
    
    final_model = Pipeline([
        ('scaler', StandardScaler()),
        ('logreg', LogisticRegression(**final_params))
    ])
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model.fit(X_train, y_train)
    
    print("   Final model trained successfully.")
        
    train_auc = roc_auc_score(y_train, final_model.predict_proba(X_train)[:, 1])
    val_auc = roc_auc_score(y_val, final_model.predict_proba(X_val)[:, 1])
    test_auc = roc_auc_score(data['y_test'], final_model.predict_proba(data['X_test'])[:, 1])
    
    print(f"   [model] Train AUC: {train_auc:.4f} | Val AUC: {val_auc:.4f} | Test AUC: {test_auc:.4f}")

    model_path = cache_dir / 'logreg_model.pkl'
    joblib.dump(final_model, model_path)
    print(f"\n=== Model saved to {model_path} ===")

if __name__ == "__main__":
    train_and_optimize()
