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

# --- Number of samples to explain in efficiency experiments ---
N_EXPLAIN = 200

# --- Results output directory ---
RESULTS_DIR = "results"
