#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

echo "============================================"
echo " CTE Pipeline"
echo "============================================"

CACHE_DIR="data/cache"

echo ""
if [ ! -f "$CACHE_DIR/xgb_model.pkl" ] || \
   [ ! -f "$CACHE_DIR/bg_truth_full.parquet" ] || \
   [ ! -f "$CACHE_DIR/bg_cte_10.parquet" ] || \
   [ ! -f "$CACHE_DIR/bg_cte_50.parquet" ] || \
   [ ! -f "$CACHE_DIR/kt_build_times.json" ]; then
    echo "[0/4] Cache niekompletny - buduje model + backgrounds (jednorazowe, ~5-10 min)"
    python experiments/prepare_cache.py
else
    echo "[0/4] Cache kompletny - pomijam prepare_cache.py"
fi

echo ""
echo "[1/4] RQ1 - Efficiency (CTE vs Random, fidelity)"
python experiments/rq1_efficiency.py

echo ""
echo "[2/4] RQ2 - Temporal drift of explanations"
python experiments/rq2_drift.py

echo ""
echo "[3/4] RQ3 - Train vs Validation explanations"
python experiments/rq3_train_val.py

echo ""
echo "[4/5] RQ4 - Break-even analysis (KernelExplainer, ~10 min)"
python experiments/rq4_breakeven.py

echo ""
echo "[5/5] Generating plots"
python evaluation/visualize.py

echo ""
echo "============================================"
echo " Done. Wyniki: results/"
echo "============================================"
