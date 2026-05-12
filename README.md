# Compress Then Explain (CTE) - Ecom Offers Scale-up

This repository contains the scalable implementation of the **Compress Then Explain (CTE)** paradigm using Kernel Thinning, specifically applied to the massive Kaggle `acquire-valued-shoppers-challenge` (Ecom-Offers) dataset.

## 🚀 Getting Started

We have already performed the memory-intensive downsampling and pre-processing of the original 22GB dataset. The cleaned, lightweight `.parquet` and `.npy` files are included in the `data/ecom-offers/` directory. 
**You do NOT need to download the raw Kaggle data or run the preprocessing script.**

### 1. Installation
Ensure you have Python installed. Install the dependencies using:
```bash
pip install -r requirements.txt
```

### 2. Running the Experiments
The repository contains 3 automated, end-to-end experiment scripts that train an XGBoost model and execute the CTE SHAP evaluation.

**RQ1: Efficiency vs. Accuracy Trade-off**
Evaluates the correlation between SHAP Ground Truth and CTE/Random backgrounds across various sizes.
```bash
python experiments/rq1_efficiency.py
```

**RQ2: Explanations Over Time (Concept Drift)**
Splits the validation data into chronological windows to test how well CTE captures temporal feature importance shifts.
```bash
python experiments/rq2_drift.py
```

**RQ3: Train vs. Validation Discrepancy**
Investigates the differences in feature reliance when the model predicts on known (Train) versus unknown (Validation) distributions.
```bash
python experiments/rq3_train_val.py
```

### 3. Visualizing Results
After running the experiments, generate publication-ready plots:
```bash
python evaluation/visualize.py
```
The output `.png` charts will be saved in the `results/` folder.
