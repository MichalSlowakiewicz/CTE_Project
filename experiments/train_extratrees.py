import joblib
import sys
from pathlib import Path
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import load_dataset

def train_and_cache_extratrees():
    print("=== Training ExtraTreesClassifier ===")
    cache_dir = Path(__file__).parent.parent / 'data' / 'cache_full'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    data = load_dataset(data_dir=Path(__file__).parent.parent / 'data' / 'ecom-offers')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    # Train a simple, default model
    model = ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    print(f"Validation AUC: {auc:.4f}")
    
    model_path = cache_dir / 'extratrees_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_cache_extratrees()
