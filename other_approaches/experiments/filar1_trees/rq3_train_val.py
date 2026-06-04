import json
import shap
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.data_loader import load_dataset
from evaluation.metrics import mae_shap

def run_filar1_rq3():
    print("=== Filar 1 (TreeExplainer) - RQ3: Train vs Val Discrepancy ===")
    
    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent / 'results' / 'filar1_trees'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = joblib.load(cache_dir / 'xgb_model.pkl')
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    
    X_train, X_val = data['X_train'], data['X_val']
    
    X_train_explain = X_train.sample(min(200, len(X_train)), random_state=42).reset_index(drop=True)
    X_val_explain = X_val.sample(min(200, len(X_val)), random_state=42).reset_index(drop=True)
    
    print("Loading Ground Truth Background (Train)...")
    bg_truth = pd.read_parquet(cache_dir / 'bg_truth_full.parquet')
    
    explainer_truth = shap.TreeExplainer(model, data=bg_truth, model_output="probability", feature_perturbation="interventional")
    shap_truth_train = explainer_truth.shap_values(X_train_explain)
    shap_truth_val = explainer_truth.shap_values(X_val_explain)
    
    bg_sizes = [10, 50, 100, 200]
    results = []
    
    for size in tqdm(bg_sizes, desc="RQ3 (Train vs Val) sizes"):
        bg_cte = pd.read_parquet(cache_dir / f'bg_cte_{size}.parquet')
        explainer_cte = shap.TreeExplainer(model, data=bg_cte, model_output="probability", feature_perturbation="interventional")
        
        shap_cte_train = explainer_cte.shap_values(X_train_explain)
        shap_cte_val = explainer_cte.shap_values(X_val_explain)
        
        res = {
            "size": size,
            "mae_train_set": float(mae_shap(shap_truth_train, shap_cte_train)),
            "mae_val_set": float(mae_shap(shap_truth_val, shap_cte_val))
        }
        results.append(res)
        
    with open(out_dir / 'rq3.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/filar1_trees/rq3.json")

if __name__ == "__main__":
    run_filar1_rq3()
