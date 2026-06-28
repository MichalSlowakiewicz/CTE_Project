# Compress Then Explain (CTE) 🌲⚡

A research repository exploring the scalability, efficiency, and robustness of the **Compress Then Explain (CTE)** algorithm against traditional Random Sampling background distributions for Explainable AI (XAI).

> **🎓 Final Research Paper:** The complete findings of this project are detailed in our main paper: [**CTE_report_paper.pdf**](./CTE_report_paper.pdf)

## 🧠 Project Overview

This project dynamically compresses large background datasets ($N=100,000$) into highly representative, ultra-small *coresets* ($N \in \{4 \dots 256\}$) using the **Kernel Thinning** algorithm. We evaluate the approximation accuracy (Global & Local MAE), temporal drift tracking, generalization gaps, and architectural agnosticism of the resulting SHAP explanations.

### Key Features
- **Data Compression:** Uses `goodpoints` to compress massive datasets via Maximum Mean Discrepancy (MMD) minimization.
- **Oracle Evaluation Pipeline:** Fully optimized pipeline utilizing C++ native `shap.TreeExplainer` and `shap.LinearExplainer` to establish exact Ground Truth baselines.
- **Architecture Agnosticism:** Benchmarks compression effectiveness across 8 diverse models belonging to two distinct algorithmic families:
  - *Tree-based Ensembles:* XGBoost, LightGBM, CatBoost, Random Forest
  - *Linear Models:* Logistic Regression, Ridge, Lasso, SVC

---

## 📂 Repository Structure

```text
CTE_Project/
├── CTE_report_paper.pdf # The main research paper
├── cte_compression/           # Core algorithm implementation
│   └── kernel_thinning.py     # Halving and Kernel Thinning logic
├── data/                      # Dataset loaders and cache
│   └── data_loader.py         # Temporal split & preprocessing for E-com data
├── experiments/               # Training and evaluation scripts
│   ├── main_pipeline/         # The main evaluation pipeline
│   │   ├── train_*.py         # Scripts to train the 8 model architectures
│   │   ├── rq1_efficiency.py  # RQ1: Global & Local MAE calculation
│   │   ├── rq2_drift.py       # RQ2: Temporal drift over 6 test periods
│   │   ├── rq3_train_val.py   # RQ3: Generalization gap
│   │   ├── plot_results.py    # Generates final visual plots
│   │   └── plot_aggregated.py # Generates the master RQ4 visualization
├── results/                   # Generated JSON metrics and PNG charts
│   └── main_pipeline/         # Outputs organized by model (e.g., xgb_tree/)
├── requirements.txt           # Python dependencies
└── run_pipeline.sh            # End-to-end execution script
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
python -m venv venv_linux
source venv_linux/bin/activate
pip install -r requirements.txt
```

### 2. Generate Results & Visualizations
The entire evaluation pipeline can be reproduced by executing the analytical scripts in the `experiments/main_pipeline/` directory, or by running the main pipeline script:

```bash
# Run the complete pipeline end-to-end
bash run_pipeline.sh
```
Or manually:
```bash
# Example: Evaluate Efficiency (RQ1)
python experiments/main_pipeline/rq1_efficiency.py

# Generate Visualizations
python experiments/main_pipeline/plot_results.py
python experiments/main_pipeline/plot_aggregated.py
```

## 📈 Results Structure
All experimental logs and computed metrics are saved in the `results/main_pipeline/` directory. 

Inside each model's directory (e.g., `results/main_pipeline/xgb_tree/`), you will find:
- `rq1_plot.png` & `rq1_table.md` - Global and local MAE bar charts and delta improvements.
- `rq2_mae_plot.png` - Temporal drift divergence (Global MAE).
- `rq2_features_plot.png` - Precise tracking of the Top 5 features across 6 time periods.
- `rq3_plot.png` - Generalization gap bar chart with Train vs Validation multipliers.

**Aggregated Findings:**
- `results/main_pipeline/rq1_aggregated.png` - Cross-model aggregated visualization proving CTE's architecture-agnostic superiority. Averages performance across Tree and Linear model families.
- `results/main_pipeline/quantitative_summary.md` - Aggregated hard percentage metrics across all models, ready for inclusion in academic reports.

---
*Developed by Okniński, Słowakiewicz, and Woźniak for rigorous research into scalable and sustainable Explainable AI (XAI).*
