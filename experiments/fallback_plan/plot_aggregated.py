"""
plot_aggregated.py — Cross-model aggregated visualizations for all RQs.

Generates:
  - rq1_aggregated.png  (Architecture Agnosticism: Tree vs Linear MAE curves)
  - rq2_aggregated.png  (Temporal Drift: averaged MAE + CTE vs Random heatmap)
  - rq3_aggregated.png  (Feature Drift: train vs val shift + CTE/Random tracking)
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

TREE_MODELS = ['xgb_tree', 'rf_tree', 'lgb_tree', 'catboost_tree']
LINEAR_MODELS = ['logreg_linear', 'ridge_linear', 'lasso_linear', 'svc_linear']
ALL_MODELS = TREE_MODELS + LINEAR_MODELS
MAX_SIZE = 256  # Cap at sqrt(n)

C_CTE = '#c44e52'
C_RAND = '#4c72b0'

PRETTY_NAMES = {
    'xgb_tree': 'XGBoost', 'rf_tree': 'Random Forest',
    'lgb_tree': 'LightGBM', 'catboost_tree': 'CatBoost',
    'logreg_linear': 'LogReg', 'ridge_linear': 'Ridge',
    'lasso_linear': 'Lasso', 'svc_linear': 'SVC',
}


def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ─────────────────────────────────────────────────────────────────────
# RQ1 AGGREGATED — Architecture Agnosticism (Tree vs Linear)
# ─────────────────────────────────────────────────────────────────────

def aggregate_rq1(base_dir, model_list):
    """Average MAE across models in a family, returning arrays + error bars."""
    all_exps = []
    for name in model_list:
        data = load_json(base_dir / name / 'rq1.json')
        if data:
            filtered = [e for e in data['experiments'] if e['size'] <= MAX_SIZE]
            all_exps.append(filtered)
    if not all_exps:
        return None

    n_models = len(all_exps)
    sizes = [e['size'] for e in all_exps[0]]
    n_sizes = len(sizes)

    has_std = 'mae_global_mean' in all_exps[0][0]['random']

    cte_g = np.zeros(n_sizes)
    cte_l = np.zeros(n_sizes)
    rand_g = np.zeros(n_sizes)
    rand_l = np.zeros(n_sizes)
    rand_g_std = np.zeros(n_sizes)
    rand_l_std = np.zeros(n_sizes)

    for exps in all_exps:
        for i, e in enumerate(exps):
            cte_g[i] += e['cte']['mae_global']
            cte_l[i] += e['cte']['mae_local']
            if has_std:
                rand_g[i] += e['random']['mae_global_mean']
                rand_l[i] += e['random']['mae_local_mean']
                rand_g_std[i] += e['random']['mae_global_std']
                rand_l_std[i] += e['random']['mae_local_std']
            else:
                rand_g[i] += e['random']['mae_global']
                rand_l[i] += e['random']['mae_local']

    return {
        'sizes': sizes,
        'cte_g': cte_g / n_models, 'cte_l': cte_l / n_models,
        'rand_g': rand_g / n_models, 'rand_l': rand_l / n_models,
        'rand_g_std': rand_g_std / n_models, 'rand_l_std': rand_l_std / n_models,
    }


def plot_rq1_aggregated(base_dir):
    tree = aggregate_rq1(base_dir, TREE_MODELS)
    linear = aggregate_rq1(base_dir, LINEAR_MODELS)
    if not tree or not linear:
        print("Skipping RQ1 aggregated — missing data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    def panel(ax, sizes, cte, rand, rand_std, title, ylabel, legend=False):
        ax.plot(sizes, cte, marker='o', color=C_CTE, lw=2, label='CTE')
        ax.plot(sizes, rand, marker='s', color=C_RAND, lw=2, label='Random (mean)')
        if np.any(rand_std > 0):
            ax.fill_between(sizes, np.maximum(rand - rand_std, 0), rand + rand_std,
                            alpha=0.2, color=C_RAND, label='Random ±1σ')
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_xlabel('Background Size (N)')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log', base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels(sizes)
        ax.grid(True, alpha=0.3)
        if legend:
            ax.legend(fontsize=9)

    panel(axes[0, 0], tree['sizes'], tree['cte_g'], tree['rand_g'], tree['rand_g_std'],
          'A) Tree Models: Global MAE', 'Global MAE', legend=True)
    panel(axes[0, 1], tree['sizes'], tree['cte_l'], tree['rand_l'], tree['rand_l_std'],
          'B) Tree Models: Local MAE', 'Local MAE')
    panel(axes[1, 0], linear['sizes'], linear['cte_g'], linear['rand_g'], linear['rand_g_std'],
          'C) Linear Models: Global MAE', 'Global MAE')
    panel(axes[1, 1], linear['sizes'], linear['cte_l'], linear['rand_l'], linear['rand_l_std'],
          'D) Linear Models: Local MAE', 'Local MAE')

    fig.suptitle('RQ1 Aggregated: CTE vs Random — Architecture Agnosticism\n'
                 f'(N ≤ {MAX_SIZE}, averaged across model families)',
                 fontsize=16, fontweight='bold', y=0.97)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = base_dir / 'rq1_aggregated.png'
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"  ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────
# RQ2 AGGREGATED — Temporal Drift across all models
# ─────────────────────────────────────────────────────────────────────

def plot_rq2_aggregated(base_dir):
    """Panel A: averaged MAE per period.  Panel B: CTE vs Random Δ% heatmap."""

    model_names = []
    model_mae_cte = []
    model_mae_rand = []
    period_labels = None

    for name in ALL_MODELS:
        data = load_json(base_dir / name / 'rq2.json')
        if not data:
            continue

        windows = data['windows']
        new_format = 'cte_mae_global' in windows[0]

        maes_cte = []
        maes_rand = []
        for w in windows:
            if new_format:
                maes_cte.append(w['cte_mae_global'])
                maes_rand.append(w['rand_mae_global_mean'])
            else:
                t = np.array(w['truth_importance'])
                c = np.array(w['cte_importance'])
                r = np.array(w.get('rand_importance_mean', w.get('rand_importance', [])))
                maes_cte.append(np.mean(np.abs(t - c)))
                maes_rand.append(np.mean(np.abs(t - r)))

        model_names.append(name)
        model_mae_cte.append(maes_cte)
        model_mae_rand.append(maes_rand)

        if period_labels is None:
            if new_format:
                period_labels = [str(w['period']) for w in windows]
            else:
                period_labels = [f"W{w.get('window', i+1)}" for i, w in enumerate(windows)]

    if not model_names:
        print("Skipping RQ2 aggregated — no data")
        return

    arr_cte = np.array(model_mae_cte)    # (n_models, n_windows)
    arr_rand = np.array(model_mae_rand)
    n_windows = arr_cte.shape[1]

    avg_cte = arr_cte.mean(axis=0)
    avg_rand = arr_rand.mean(axis=0)
    std_cte = arr_cte.std(axis=0)
    std_rand = arr_rand.std(axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: averaged MAE over periods
    x = list(range(n_windows))
    ax1.plot(x, avg_cte, marker='o', color=C_CTE, lw=2, label='CTE (avg)')
    ax1.fill_between(x, avg_cte - std_cte, avg_cte + std_cte,
                     alpha=0.15, color=C_CTE)
    ax1.plot(x, avg_rand, marker='s', color=C_RAND, lw=2, label='Random (avg)')
    ax1.fill_between(x, avg_rand - std_rand, avg_rand + std_rand,
                     alpha=0.15, color=C_RAND)
    ax1.set_title('A) Avg. Global MAE per Time Period', loc='left', fontweight='bold')
    ax1.set_xlabel('Time Period')
    ax1.set_ylabel('Global MAE (avg. across 8 models)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(period_labels)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel B: heatmap — CTE Δ% vs Random
    # Value = (rand - cte) / rand  →  positive = CTE better, negative = Random better
    delta = (arr_rand - arr_cte) / (arr_rand + 1e-12) * 100
    pretty = [PRETTY_NAMES.get(n, n) for n in model_names]

    im = ax2.imshow(delta, aspect='auto', cmap='RdYlGn', vmin=-50, vmax=50)
    ax2.set_xticks(range(n_windows))
    ax2.set_xticklabels(period_labels)
    ax2.set_yticks(range(len(pretty)))
    ax2.set_yticklabels(pretty)
    ax2.set_title('B) CTE vs Random: Δ% per Model & Period', loc='left', fontweight='bold')
    ax2.set_xlabel('Time Period')

    # Annotate cells
    for i in range(len(model_names)):
        for j in range(n_windows):
            val = delta[i, j]
            color = 'white' if abs(val) > 30 else 'black'
            sign = '+' if val > 0 else ''
            ax2.text(j, i, f'{sign}{val:.0f}%', ha='center', va='center',
                     fontsize=8, color=color, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Δ% (positive = CTE better, negative = Random better)')
    fig.suptitle('RQ2 Aggregated: Temporal Feature Importance Tracking (N=128)',
                 fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    out = base_dir / 'rq2_aggregated.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────
# RQ3 AGGREGATED — Train-Val Feature Drift across all models
# ─────────────────────────────────────────────────────────────────────

def plot_rq3_aggregated(base_dir):
    """Panel A: CTE vs Random MAE on train vs val explained samples.
       Panel B: Top 10 most drifted features (pooled across models)."""

    # Collect per-model data
    model_names = []
    cte_train_maes = []
    cte_val_maes = []
    rand_train_maes = []
    rand_val_maes = []
    rand_train_stds = []
    rand_val_stds = []

    # Aggregate feature drift across models
    all_drift = {}  # feature_name → list of drift values

    new_format_detected = False

    for name in ALL_MODELS:
        data = load_json(base_dir / name / 'rq3.json')
        if not data:
            continue

        # Detect format
        if 'drift_ranking' in data:
            new_format_detected = True
            model_names.append(name)
            cte_train_maes.append(data['cte']['train_mae'])
            cte_val_maes.append(data['cte']['val_mae'])
            rand_train_maes.append(data['random']['train_mae_mean'])
            rand_val_maes.append(data['random']['val_mae_mean'])
            rand_train_stds.append(data['random']['train_mae_std'])
            rand_val_stds.append(data['random']['val_mae_std'])

            # Collect feature drift
            feat_names = data['feature_names']
            for i, fname in enumerate(feat_names):
                if fname not in all_drift:
                    all_drift[fname] = []
                all_drift[fname].append(data['feature_drift'][i])

    if not new_format_detected:
        # Fall back to old format
        _plot_rq3_aggregated_old(base_dir)
        return

    if not model_names:
        print("Skipping RQ3 aggregated — no data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Grouped bar chart — CTE vs Random MAE on Train vs Val samples
    pretty = [PRETTY_NAMES.get(n, n) for n in model_names]
    y = np.arange(len(pretty))
    height = 0.2

    # For each model: 4 bars (CTE-train, CTE-val, Rand-train, Rand-val)
    ax1.barh(y - 1.5*height, cte_train_maes, height, label='CTE (train samples)',
             color=C_CTE, alpha=0.7)
    ax1.barh(y - 0.5*height, cte_val_maes, height, label='CTE (val samples)',
             color=C_CTE, alpha=1.0)
    # Clip error bars so they don't go below 0 (MAE ≥ 0)
    rt_arr, rt_std = np.array(rand_train_maes), np.array(rand_train_stds)
    rv_arr, rv_std = np.array(rand_val_maes), np.array(rand_val_stds)
    rt_err_lo = np.minimum(rt_std, rt_arr)  # can't go below 0
    rv_err_lo = np.minimum(rv_std, rv_arr)
    ax1.barh(y + 0.5*height, rand_train_maes, height, label='Random (train samples)',
             color=C_RAND, alpha=0.7, xerr=[rt_err_lo, rt_std], capsize=2)
    ax1.barh(y + 1.5*height, rand_val_maes, height, label='Random (val samples)',
             color=C_RAND, alpha=1.0, xerr=[rv_err_lo, rv_std], capsize=2)
    ax1.set_yticks(y)
    ax1.set_yticklabels(pretty)
    ax1.invert_yaxis()
    ax1.set_xlabel('Global MAE vs Full Background')
    ax1.set_title('A) CTE vs Random: Train vs Val Context', loc='left', fontweight='bold')
    ax1.grid(True, axis='x', alpha=0.3)
    ax1.legend(fontsize=8, loc='lower right')

    # Panel B: Top 10 most drifted features (averaged across models)
    avg_drift = {fn: np.mean(vals) for fn, vals in all_drift.items()}
    sorted_features = sorted(avg_drift.keys(), key=lambda k: avg_drift[k], reverse=True)[:10]

    feat_labels = sorted_features
    drift_values = [avg_drift[f] for f in sorted_features]

    y2 = np.arange(len(feat_labels))
    bars = ax2.barh(y2, drift_values, height=0.6, color='#55a868')
    ax2.set_yticks(y2)
    ax2.set_yticklabels(feat_labels)
    ax2.invert_yaxis()
    ax2.set_xlabel('Avg |Δ Importance| (Train → Val)')
    ax2.set_title('B) Top 10 Most Drifted Features', loc='left', fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3)

    # Annotate
    max_d = max(drift_values) if drift_values else 1
    for bar, val in zip(bars, drift_values):
        ax2.text(bar.get_width() + max_d * 0.02, bar.get_y() + bar.get_height() / 2,
                 f'{val:.2e}', va='center', fontsize=8)

    fig.suptitle('RQ3 Aggregated: Train vs Validation Feature Drift (N=128)',
                 fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    out = base_dir / 'rq3_aggregated.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


def _plot_rq3_aggregated_old(base_dir):
    """Fallback for old RQ3 JSON format (baseline gap only)."""
    model_gaps = {}
    all_train_imp = []
    all_val_imp = []

    for name in ALL_MODELS:
        data = load_json(base_dir / name / 'rq3.json')
        if not data:
            continue
        train_imp = np.array(data['train']['truth_importance'])
        val_imp = np.array(data['val']['truth_importance'])
        model_gaps[name] = data['baseline_diff_train_val']['global_mae']
        top_idx = np.argsort(train_imp)[-10:]
        all_train_imp.extend(train_imp[top_idx].tolist())
        all_val_imp.extend(val_imp[top_idx].tolist())

    if not model_gaps:
        print("Skipping RQ3 aggregated (old format) — no data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.scatter(all_train_imp, all_val_imp, alpha=0.4, s=30, color=C_CTE)
    lim_max = max(max(all_train_imp), max(all_val_imp)) * 1.1
    ax1.plot([0, lim_max], [0, lim_max], 'k--', lw=1, alpha=0.5, label='y = x')
    ax1.set_xlabel('Train Importance')
    ax1.set_ylabel('Val Importance')
    ax1.set_title('A) Train vs Val Feature Importance', loc='left', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    names_sorted = sorted(model_gaps.keys(), key=lambda k: model_gaps[k], reverse=True)
    pretty = [PRETTY_NAMES.get(n, n) for n in names_sorted]
    gaps = [model_gaps[n] for n in names_sorted]
    colors = [C_CTE if n in TREE_MODELS else C_RAND for n in names_sorted]
    y = np.arange(len(pretty))
    ax2.barh(y, gaps, color=colors, height=0.6)
    ax2.set_yticks(y)
    ax2.set_yticklabels(pretty)
    ax2.invert_yaxis()
    ax2.set_xlabel('Baseline Gap: Global MAE(Train vs Val)')
    ax2.set_title('B) Generalization Gap per Model', loc='left', fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3)

    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(facecolor=C_CTE, label='Tree'), Patch(facecolor=C_RAND, label='Linear')],
               loc='lower right', fontsize=9)
    fig.suptitle('RQ3 Aggregated: Train-Validation Gap', fontsize=16, fontweight='bold', y=1.0)
    plt.tight_layout()
    out = base_dir / 'rq3_aggregated.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {out.name}")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent / "results" / "fallback_plan"
    print("Generating aggregated cross-model plots...")
    plot_rq1_aggregated(base_dir)
    plot_rq2_aggregated(base_dir)
    plot_rq3_aggregated(base_dir)
    print("Done!")
