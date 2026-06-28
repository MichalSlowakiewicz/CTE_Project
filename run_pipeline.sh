#!/bin/bash
set -e

echo "=== Starting Full CTE Pipeline ==="

PYTHON="venv_linux/bin/python3.11"

echo "1. Generating CTE Coresets..."
$PYTHON experiments/prepare_cache_full.py

echo "2. Running RQ1 (Efficiency)..."
$PYTHON experiments/main_pipeline/rq1_efficiency.py

echo "3. Running RQ2 (Concept Drift)..."
$PYTHON experiments/main_pipeline/rq2_drift.py

echo "4. Running RQ3 (Train vs Val)..."
$PYTHON experiments/main_pipeline/rq3_train_val.py

echo "5. Running RQ4 (Break-even)..."
$PYTHON experiments/main_pipeline/rq4_breakeven.py

echo "6. Generating Plots and Summaries..."
$PYTHON experiments/main_pipeline/plot_results.py
$PYTHON experiments/main_pipeline/plot_aggregated.py
$PYTHON experiments/main_pipeline/generate_summary_metrics.py

echo "=== Pipeline Completed Successfully! ==="
