# Coreset Tree Explainer (CTE) 🌲⚡

A research repository exploring the scalability, efficiency, and robustness of the **Coreset Tree Explainer (CTE)** algorithm against traditional Random Sampling background distributions for Explainable AI (XAI).

## 🧠 Project Overview
This project dynamically compresses large background datasets ($N=100,000$) into highly representative, ultra-small *coresets* ($N \in \{4 \dots 1024\}$) using the **Kernel Thinning** algorithm. We evaluate the approximation accuracy (Global & Local MAE), temporal drift tracking, generalization gaps, and architectural agnosticism of the resulting SHAP explanations.

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
├── cte_compression/           # Core algorithm implementation
│   └── kernel_thinning.py     # Halving and Kernel Thinning logic
├── data/                      # Dataset loaders and cache
│   └── data_loader.py         # Temporal split & preprocessing for E-com data
├── experiments/               # Training and evaluation scripts
│   ├── fallback_plan/         # The core v15 optimized evaluation pipeline
│   │   ├── train_*.py         # Scripts to train the 8 model architectures
│   │   ├── rq1_efficiency.py  # RQ1: Global & Local MAE calculation
│   │   ├── rq2_drift.py       # RQ2: Temporal drift over 5 windows
│   │   ├── rq3_train_val.py   # RQ3: Generalization gap
│   │   ├── plot_results.py    # Generates final visual plots
│   │   └── plot_aggregated.py # Generates the master RQ4 visualization
├── results/                   # Generated JSON metrics and PNG charts
│   └── fallback_plan/         # Outputs organized by model (e.g., xgb_tree/)
├── academic_discussion.md     # In-depth analysis of scientific findings
└── requirements.txt           # Python dependencies
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
The entire evaluation pipeline can be reproduced by executing the analytical scripts in the `experiments/fallback_plan/` directory.

```bash
# Example: Evaluate Efficiency (RQ1)
python experiments/fallback_plan/rq1_efficiency.py

# Generate Visualizations
python experiments/fallback_plan/plot_results.py
python experiments/fallback_plan/plot_aggregated.py
```

## 📈 Results Structure
All experimental logs and computed metrics are saved in the `results/fallback_plan/` directory. 

Inside each model's directory (e.g., `results/fallback_plan/xgb_tree/`), you will find:
- `rq1_plot.png` & `rq1_table.md` - Global and local MAE bar charts and delta improvements.
- `rq2_mae_plot.png` - Temporal drift divergence (Global MAE).
- `rq2_features_plot.png` - Precise tracking of the Top 5 features across 5 time windows.
- `rq3_plot.png` - Generalization gap bar chart with Train vs Validation multipliers.

**Aggregated Findings:**
- `results/fallback_plan/rq4_aggregated.png` - The final, overarching visualization proving CTE's architecture-agnostic superiority. It averages the performance across all Tree and Linear models, decisively concluding the research project.
- `results/fallback_plan/quantitative_summary.md` - Aggregated hard percentage metrics across all models, ready for inclusion in academic reports.

---
*Developed for rigorous research into scalable and sustainable Explainable AI (XAI).*
