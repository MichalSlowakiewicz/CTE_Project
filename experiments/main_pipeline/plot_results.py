"""
plot_results.py — Per-model visualizations for RQ1, RQ2, and RQ3.

Generates per-model plots in results/main_pipeline/<model_name>/:
  - rq1_plot.png    + rq1_table.md
  - rq2_mae_plot.png + rq2_features_plot.png
  - rq3_plot.png
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

MAX_SIZE = 256  # = √n' for M≈100k: the native Compress++ output size; larger targets cannot be produced

def plot_rq1(model_dir, data):
    exps = [e for e in data['experiments'] if e['size'] <= MAX_SIZE]
    sizes = [exp['size'] for exp in exps]

    
    # Support both old format (random.mae_global) and new format (random.mae_global_mean)
    has_std = 'mae_global_mean' in exps[0]['random']
    has_cte_std = 'mae_global_mean' in exps[0]['cte']
    
    if has_cte_std:
        cte_global = [exp['cte']['mae_global_mean'] for exp in exps]
        cte_global_std = [exp['cte']['mae_global_std'] for exp in exps]
        cte_local = [exp['cte']['mae_local_mean'] for exp in exps]
        cte_local_std = [exp['cte']['mae_local_std'] for exp in exps]
    else:
        cte_global = [exp['cte']['mae_global'] for exp in exps]
        cte_global_std = [0] * len(sizes)
        cte_local = [exp['cte']['mae_local'] for exp in exps]
        cte_local_std = [0] * len(sizes)

    if has_std:
        rand_global = [exp['random']['mae_global_mean'] for exp in exps]
        rand_global_std = [exp['random']['mae_global_std'] for exp in exps]
        rand_local = [exp['random']['mae_local_mean'] for exp in exps]
        rand_local_std = [exp['random']['mae_local_std'] for exp in exps]
    else:
        rand_global = [exp['random']['mae_global'] for exp in exps]
        rand_global_std = [0] * len(sizes)
        rand_local = [exp['random']['mae_local'] for exp in exps]
        rand_local_std = [0] * len(sizes)
    
    # 1. Generate the table as a markdown file for the walkthrough
    table_lines = ["| N_bg | CTE Global MAE | Rand Global MAE | Δ Improvement | CTE Local MAE | Rand Local MAE |"]
    table_lines.append("|---|---|---|---|---|---|")
    for i, s in enumerate(sizes):
        cg, rg, cl, rl = cte_global[i], rand_global[i], cte_local[i], rand_local[i]
        imp = ((rg - cg) / rg) * 100 if rg > 0 else 0
        imp_str = f"**+{imp:.0f}%**" if imp > 0 else f"{imp:.0f}%"
        
        if has_cte_std:
            cg_str = f"{cg:.2e}±{cte_global_std[i]:.2e}"
            cl_str = f"{cl:.2e}±{cte_local_std[i]:.2e}"
        else:
            cg_str = f"{cg:.2e}"
            cl_str = f"{cl:.2e}"

        if has_std:
            rg_str = f"{rg:.2e}±{rand_global_std[i]:.2e}"
            rl_str = f"{rl:.2e}±{rand_local_std[i]:.2e}"
        else:
            rg_str = f"{rg:.2e}"
            rl_str = f"{rl:.2e}"
        
        table_lines.append(f"| {s} | {cg_str} | {rg_str} | {imp_str} | {cl_str} | {rl_str} |")
    
    with open(model_dir / "rq1_table.md", "w") as f:
        f.write("\n".join(table_lines))
        
    # 2. Generate Horizontal Bar Charts with error bars
    s_sub = sizes
    
    cg_arr = np.array(cte_global)
    cl_arr = np.array(cte_local)
    cg_std = np.array(cte_global_std)
    cl_std = np.array(cte_local_std)
    rg_arr = np.array(rand_global)
    rl_arr = np.array(rand_local)
    rg_std = np.array(rand_global_std)
    rl_std = np.array(rand_local_std)
    
    # Scale for readability (x10^-5)
    scale = 1e5
    
    y = np.arange(len(s_sub))
    height = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    cg_err_lo = np.minimum(cg_std, cg_arr) * scale  # clip at 0
    ax1.barh(y - height/2, cg_arr * scale, height, label='CTE', color='#c44e52',
             xerr=[cg_err_lo, cg_std * scale], capsize=3)
    rg_err_lo = np.minimum(rg_std, rg_arr) * scale  # clip at 0
    ax1.barh(y + height/2, rg_arr * scale, height, label='random', color='#4c72b0',
             xerr=[rg_err_lo, rg_std * scale], capsize=3)
    ax1.set_yticks(y)
    ax1.set_yticklabels([f"N={s}" for s in s_sub])
    ax1.invert_yaxis()
    ax1.set_xlabel('Global MAE ($\\times 10^{-5}$)')
    ax1.set_title('A) Global Feature Importance', loc='left', fontweight='bold')
    ax1.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right')
    
    cl_err_lo = np.minimum(cl_std, cl_arr) * scale  # clip at 0
    ax2.barh(y - height/2, cl_arr * scale, height, label='CTE', color='#c44e52',
             xerr=[cl_err_lo, cl_std * scale], capsize=3)
    rl_err_lo = np.minimum(rl_std, rl_arr) * scale  # clip at 0
    ax2.barh(y + height/2, rl_arr * scale, height, label='random', color='#4c72b0',
             xerr=[rl_err_lo, rl_std * scale], capsize=3)
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"N={s}" for s in s_sub])
    ax2.invert_yaxis()
    ax2.set_xlabel('Local MAE ($\\times 10^{-5}$)')
    ax2.set_title('B) Per-Sample Explanation', loc='left', fontweight='bold')
    ax2.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(model_dir / 'rq1_plot.png', dpi=300)
    plt.close()

def plot_rq2(model_dir, data):
    """Plot RQ2 — supports both old (5-window) and new (6-period) JSON format."""
    windows = data['windows']
    
    # Detect new format (has pre-computed MAE/RE fields)
    new_format = 'cte_mae_global' in windows[0] or 'cte_mae_global_mean' in windows[0]
    
    if new_format:
        # New format: period-based windows with pre-computed metrics
        periods = [w['period'] for w in windows]
        x_labels = [str(p) for p in periods]
        x_ticks = list(range(len(periods)))
        
        has_cte_std = 'cte_mae_global_mean' in windows[0]
        if has_cte_std:
            mae_cte = [w['cte_mae_global_mean'] for w in windows]
            mae_cte_std = [w['cte_mae_global_std'] for w in windows]
        else:
            mae_cte = [w.get('cte_mae_global', w.get('cte_mae_global_mean', 0)) for w in windows]
            mae_cte_std = []

        mae_rand = [w['rand_mae_global_mean'] for w in windows]
        mae_rand_std = [w['rand_mae_global_std'] for w in windows]
    else:
        # Old format: compute MAE from importance vectors
        x_labels = [str(w.get('window', i+1)) for i, w in enumerate(windows)]
        x_ticks = list(range(len(windows)))
        has_std = 'rand_importance_mean' in windows[0]
        
        mae_cte = []
        mae_cte_std = []
        mae_rand = []
        mae_rand_std = []
        for w in windows:
            t = np.array(w['truth_importance'])
            c = np.array(w['cte_importance'])
            r = np.array(w.get('rand_importance_mean', w.get('rand_importance', [])))
            mae_cte.append(np.mean(np.abs(t - c)))
            mae_rand.append(np.mean(np.abs(t - r)))
            if has_std:
                mae_rand_std.append(np.mean(np.array(w['rand_importance_std'])))
            if 'cte_importance_std' in w:
                mae_cte_std.append(np.mean(np.array(w['cte_importance_std'])))
    
    # --- Plot 1: Global MAE over time periods ---
    plt.figure(figsize=(8, 5))
    plt.plot(x_ticks, mae_cte, marker='o', label='CTE (N=128)', color='#c44e52', linewidth=2)
    plt.plot(x_ticks, mae_rand, marker='s', label='Random (N=128)', color='#4c72b0', linewidth=2)
    
    if mae_cte_std:
        plt.fill_between(x_ticks,
                         [m - s for m, s in zip(mae_cte, mae_cte_std)],
                         [m + s for m, s in zip(mae_cte, mae_cte_std)],
                         alpha=0.2, color='#c44e52', label='CTE ±σ')
    
    if mae_rand_std:
        plt.fill_between(x_ticks,
                         [m - s for m, s in zip(mae_rand, mae_rand_std)],
                         [m + s for m, s in zip(mae_rand, mae_rand_std)],
                         alpha=0.2, color='#4c72b0', label='Random ±σ')
    
    plt.title(f'{model_dir.name}: Feature Importance Tracking Across Time Periods')
    plt.xlabel('Time Period')
    plt.ylabel('Global MAE vs Full Background')
    plt.xticks(x_ticks, x_labels)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / 'rq2_mae_plot.png', dpi=300)
    plt.close()

    # --- Plot 2: Feature Tracking over Time ---
    feature_names = data['feature_names']
    # Find top 5 features by mean truth importance across windows
    all_truth = np.array([w['truth_importance'] for w in windows])
    mean_truth = np.mean(all_truth, axis=0)
    top_5_idx = np.argsort(mean_truth)[-5:][::-1]
    top_5_names = [feature_names[i] for i in top_5_idx]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = ['#c44e52', '#4c72b0', '#eebd52', '#55a868', '#4c4c4c']
    
    for ax, title, key, std_key, letter in [
        (axes[0], 'Ground Truth', 'truth_importance', None, 'A)'),
        (axes[1], 'CTE (N=128)', 
         'cte_importance_mean' if 'cte_importance_mean' in windows[0] else 'cte_importance', 
         'cte_importance_std' if 'cte_importance_std' in windows[0] else None, 'B)'),
        (axes[2], 'Random (N=128)', 
         'rand_importance_mean' if 'rand_importance_mean' in windows[0] else 'rand_importance',
         'rand_importance_std' if 'rand_importance_std' in windows[0] else None, 'C)'),
    ]:
        for color, fi, name in zip(colors, top_5_idx, top_5_names):
            vals = [w[key][fi] for w in windows]
            ax.plot(x_ticks, vals, marker='o', color=color, linewidth=2, label=name)
            
            if std_key:
                stds = [w[std_key][fi] for w in windows]
                ax.fill_between(x_ticks,
                                [v - s for v, s in zip(vals, stds)],
                                [v + s for v, s in zip(vals, stds)],
                                alpha=0.15, color=color)
        
        ax.set_title(f'{letter} {title}', loc='left', fontweight='bold')
        ax.set_xlabel('Time Period')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        if letter == 'A)':
            ax.set_ylabel('Mean |SHAP|')
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(model_dir / 'rq2_features_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_rq3(model_dir, data):
    """Plot RQ3 — supports both old and new JSON format."""
    feature_names = data['feature_names']
    
    # Detect new format
    new_format = 'drift_ranking' in data
    
    if new_format:
        # New format: train vs val with CTE and Random comparison
        truth_train = np.array(data['truth']['train_importance'])
        truth_val = np.array(data['truth']['val_importance'])
        
        # Top 10 features by drift magnitude
        drift_ranking = data['drift_ranking'][:10]
        
        names = [feature_names[i] for i in drift_ranking]
        t_vals = [truth_train[i] for i in drift_ranking]
        v_vals = [truth_val[i] for i in drift_ranking]
        drift_vals = [data['feature_drift'][i] for i in drift_ranking]
    else:
        # Old format
        train_imp = np.array(data['train']['truth_importance'])
        val_imp = np.array(data['val']['truth_importance'])
        top_10_idx = np.argsort(train_imp)[-10:][::-1]
        
        names = [feature_names[i] for i in top_10_idx]
        t_vals = [train_imp[i] for i in top_10_idx]
        v_vals = [val_imp[i] for i in top_10_idx]
    
    if new_format:
        cte_train_mae = data['cte']['train_mae_mean'] if 'train_mae_mean' in data['cte'] else data['cte'].get('train_mae', 0)
        cte_val_mae = data['cte']['val_mae_mean'] if 'val_mae_mean' in data['cte'] else data['cte'].get('val_mae', 0)
        cte_train_std = data['cte'].get('train_mae_std', 0)
        cte_val_std = data['cte'].get('val_mae_std', 0)
        
        rand_train_mae = data['random']['train_mae_mean']
        rand_val_mae = data['random']['val_mae_mean']
        rand_train_std = data['random']['train_mae_std']
        rand_val_std = data['random']['val_mae_std']
        
        categories = ['Train Samples', 'Val Samples']
        cte_maes = [cte_train_mae, cte_val_mae]
        cte_stds = [cte_train_std, cte_val_std]
        rand_maes = [rand_train_mae, rand_val_mae]
        rand_stds = [rand_train_std, rand_val_std]
        
        x = np.arange(len(categories))
        width = 0.35
        
        fig, ax2 = plt.subplots(figsize=(8, 7))
        
        cte_arr = np.array(cte_maes)
        cte_std_arr = np.array(cte_stds)
        yerr_cte_lo = np.minimum(cte_std_arr, cte_arr)
        
        ax2.bar(x - width/2, cte_maes, width, label='CTE', color='#c44e52',
                yerr=[yerr_cte_lo, cte_std_arr], capsize=5)
        
        rand_arr = np.array(rand_maes)
        rand_std_arr = np.array(rand_stds)
        yerr_lo = np.minimum(rand_std_arr, rand_arr)
        ax2.bar(x + width/2, rand_maes, width, label='Random (mean)', color='#4c72b0',
                yerr=[yerr_lo, rand_std_arr], capsize=5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(categories)
        ax2.set_ylabel('Global MAE vs Full Background')
        ax2.set_title('Approximation Error: Train vs Val Context', loc='left', fontweight='bold')
        ax2.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax2.legend()
    else:
        fig, ax2 = plt.subplots(figsize=(8, 7))
        ax2.text(0.5, 0.5, 'Not supported', ha='center')
    
    plt.tight_layout()
    plt.savefig(model_dir / 'rq3_plot.png', dpi=300)
    plt.close()


def main():
    base_dir = Path(__file__).parent.parent.parent / "results" / "main_pipeline"
    
    for model_dir in base_dir.iterdir():
        if not model_dir.is_dir(): continue
        
        if (model_dir / "rq1.json").exists():
            with open(model_dir / "rq1.json") as f:
                plot_rq1(model_dir, json.load(f))
                
        if (model_dir / "rq2.json").exists():
            with open(model_dir / "rq2.json") as f:
                plot_rq2(model_dir, json.load(f))
                
        if (model_dir / "rq3.json").exists():
            with open(model_dir / "rq3.json") as f:
                plot_rq3(model_dir, json.load(f))

if __name__ == "__main__":
    main()
    print("All per-model visualizations generated successfully!")
