import json
import time
import joblib
from pathlib import Path
import warnings

import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import load_dataset
from models.trainer import train_model
from cte_compression.kernel_thinning import build_cte_background, build_iid_background

def prepare_global_cache():
    print("=== CTE Global Cache Preparation ===")
    
    cache_dir = Path(__file__).parent.parent / 'data' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Loading dataset...")
    try:
        data = load_dataset(data_dir=Path(__file__).parent.parent / 'data' / 'ecom-offers')
    except Exception as e:
        warnings.warn(f"Failed to load TabReD dataset, falling back to synthetic. Error: {e}")
        data = load_dataset()
        
    train_size = min(50_000, len(data['X_train']))
    sampled_indices = data['X_train'].sample(train_size, random_state=42).index
    X_train = data['X_train'].loc[sampled_indices].reset_index(drop=True)
    y_train = data['y_train'][sampled_indices]
    
    print(f"\n2. Training XGBoost model on {train_size} samples...")
    model = train_model(X_train, y_train, X_val=data['X_val'], y_val=data['y_val'], verbose=True)
    model_path = cache_dir / 'xgb_model.pkl'
    joblib.dump(model, model_path)
    print(f"   Model saved to {model_path}")
    
    print(f"\n3. Building Ground Truth Background (Size: {len(X_train)})")
    bg_truth_path = cache_dir / 'bg_truth_full.parquet'
    X_train.to_parquet(bg_truth_path)
    print(f"   Ground Truth saved to {bg_truth_path}")
    
    bg_sizes = [10, 50, 100, 200]
    for size in bg_sizes:
        print(f"\n4. Building CTE Background (Target Size: {size})")
        t0 = time.time()
        bg_cte = build_cte_background(X_train, target_size=size, verbose=True)
        t1 = time.time()
        
        bg_cte_path = cache_dir / f'bg_cte_{size}.parquet'
        bg_cte.to_parquet(bg_cte_path)
        print(f"   CTE Background completed in {t1-t0:.1f} seconds. Saved to {bg_cte_path}")
    
    print("\n=== Global Cache Preparation Complete! ===")
    print("You can now run rq1, rq2, and rq3 scripts instantly.")

if __name__ == "__main__":
    prepare_global_cache()
