"""
config.py - Central configuration for the CTE project.

All experiment parameters live here. Change values in one place
rather than hunting through individual scripts.
"""

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Dataset ---
# Path to the preprocessed ecom_offers data directory from TabReD.
# Expected files: X_train.parquet, X_val.parquet, y_train.parquet, y_val.parquet
# (or .npy / .csv depending on the preprocessing script used).
# Set to None to fall back to the built-in synthetic dataset.
TABRED_DATA_DIR = None  # e.g. "data/ecom_offers"

# --- Background set sizes to evaluate (for RQ1 sweep) ---
# These are the N values tested for both i.i.d. and CTE backgrounds.
BACKGROUND_SIZES = [25, 50, 100, 200, 500]

# --- Default background sizes used across experiments ---
IID_BACKGROUND_SIZE = 1000   # large i.i.d. reference (pseudo ground truth)
CTE_BACKGROUND_SIZE = 50     # small kernel-thinned coreset

# --- Number of samples to explain in efficiency experiments ---
N_EXPLAIN = 200

# --- XGBoost model hyperparameters ---
MODEL_PARAMS = {
    "n_estimators": 100,
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": RANDOM_SEED,
    "eval_metric": "logloss",
}

# --- Results output directory ---
RESULTS_DIR = "results"
