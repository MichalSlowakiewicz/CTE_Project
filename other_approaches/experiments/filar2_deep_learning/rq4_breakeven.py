import time
import json
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.data_loader import load_dataset
from cte_compression.kernel_thinning import build_cte_background

from prepare_mlp import RobustMLP, compute_expected_gradients

def run_filar2_rq4():
    print("=== Filar 2 (Deep Learning) - RQ4: Break-even Point ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    out_dir = Path(__file__).parent.parent.parent / 'results' / 'filar2_deep_learning'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    X_train, X_val = data['X_train'], data['X_val']
    
    model = RobustMLP(X_train.shape[1]).to(device)
    model.load_state_dict(torch.load(cache_dir / 'mlp_model.pt', map_location=device))
    model.eval()
    
    val_explain_sizes = [10, 50, 100, 200, 500]
    bg_size = 50 
    
    print(f"Measuring CTE Compression time for target size {bg_size}...")
    t0 = time.time()
    bg_cte = build_cte_background(X_train, target_size=bg_size, verbose=False)
    compression_time = time.time() - t0
    print(f"Compression time: {compression_time:.2f} s")
    
    bg_truth_sampled = pd.read_parquet(cache_dir / 'bg_truth_full.parquet').sample(4000, random_state=42)
    
    results = []
    
    for n_val in tqdm(val_explain_sizes, desc="Explained Samples"):
        X_val_explain = X_val.sample(min(n_val, len(X_val)), random_state=42).reset_index(drop=True)
        
        t0 = time.time()
        _ = compute_expected_gradients(model, X_val_explain, bg_cte, device)
        explain_time_cte = time.time() - t0
        
        t0 = time.time()
        _ = compute_expected_gradients(model, X_val_explain, bg_truth_sampled, device)
        explain_time_truth = time.time() - t0
        
        res = {
            "num_val_samples": n_val,
            "cte": {
                "compression_time": compression_time,
                "explain_time": explain_time_cte,
                "total_time": compression_time + explain_time_cte
            },
            "truth_bg_1000": {
                "explain_time": explain_time_truth,
                "total_time": explain_time_truth
            }
        }
        results.append(res)
        
    with open(out_dir / 'rq4.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/filar2_deep_learning/rq4.json")

if __name__ == "__main__":
    run_filar2_rq4()
