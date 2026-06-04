import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.data_loader import load_dataset
from evaluation.metrics import mae_shap

from prepare_mlp import RobustMLP, compute_expected_gradients

def run_filar2_rq3():
    print("=== Filar 2 (Deep Learning) - RQ3: Train vs Val Discrepancy ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent / 'results' / 'filar2_deep_learning'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_val = data['X_train'], data['X_val']
    
    model = RobustMLP(X_train.shape[1]).to(device)
    model.load_state_dict(torch.load(cache_dir / 'mlp_model.pt', map_location=device))
    model.eval()
    
    X_train_explain = X_train.sample(min(100, len(X_train)), random_state=42).reset_index(drop=True)
    X_val_explain = X_val.sample(min(100, len(X_val)), random_state=42).reset_index(drop=True)
    
    print("Loading Ground Truth Background (Train)...")
    bg_truth_sampled = pd.read_parquet(cache_dir / 'bg_truth_full.parquet').sample(4000, random_state=42)
    
    shap_truth_train = compute_expected_gradients(model, X_train_explain, bg_truth_sampled, device)
    shap_truth_val = compute_expected_gradients(model, X_val_explain, bg_truth_sampled, device)
    
    bg_sizes = [10, 50, 100, 200]
    results = []
    
    for size in tqdm(bg_sizes, desc="RQ3 (Filar 2) sizes"):
        bg_cte = pd.read_parquet(cache_dir / f'bg_cte_{size}.parquet')
        
        shap_cte_train = compute_expected_gradients(model, X_train_explain, bg_cte, device)
        shap_cte_val = compute_expected_gradients(model, X_val_explain, bg_cte, device)
        
        res = {
            "size": size,
            "mae_train_set": float(mae_shap(shap_truth_train, shap_cte_train)),
            "mae_val_set": float(mae_shap(shap_truth_val, shap_cte_val))
        }
        results.append(res)
        
    with open(out_dir / 'rq3.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/filar2_deep_learning/rq3.json")

if __name__ == "__main__":
    run_filar2_rq3()
