import joblib
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import load_dataset

def train_and_cache_lasso():
    print("=== Training Lasso Logistic Regression (L1) ===")
    cache_dir = Path(__file__).parent.parent / 'data' / 'cache_full'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    data = load_dataset(data_dir=Path(__file__).parent.parent / 'data' / 'ecom-offers')
    X_train, y_train = data['X_train'], data['y_train']
    X_val, y_val = data['X_val'], data['y_val']
    
    # Lasso requires scaling and uses penalty='l1' with solver='liblinear' or 'saga'
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42))
    ])
    
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    
    # Count how many features were completely zeroed out by L1 penalty
    coefs = model.named_steps['lasso'].coef_[0]
    zeroed = sum(coefs == 0)
    print(f"Validation AUC: {auc:.4f}")
    print(f"Features zeroed out by Lasso: {zeroed} / {len(coefs)}")
    
    model_path = cache_dir / 'lasso_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_cache_lasso()
