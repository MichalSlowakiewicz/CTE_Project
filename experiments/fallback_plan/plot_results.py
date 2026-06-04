import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_rq1(model_dir, data):
    sizes = [exp['size'] for exp in data['experiments']]
    cte_global = [exp['cte']['mae_global'] for exp in data['experiments']]
    rand_global = [exp['random']['mae_global'] for exp in data['experiments']]
    cte_local = [exp['cte']['mae_local'] for exp in data['experiments']]
    rand_local = [exp['random']['mae_local'] for exp in data['experiments']]
    
    # 1. Generate the table as a markdown file for the walkthrough
    table_lines = ["| N_bg | CTE Global MAE | Rand Global MAE | $\\Delta$ Improvement | CTE Local MAE | Rand Local MAE |"]
    table_lines.append("|---|---|---|---|---|---|")
    for s, cg, rg, cl, rl in zip(sizes, cte_global, rand_global, cte_local, rand_local):
        imp = ((rg - cg) / rg) * 100 if rg > 0 else 0
        imp_str = f"**+{imp:.0f}%**" if imp > 0 else f"{imp:.0f}%"
        table_lines.append(f"| {s} | {cg:.2e} | {rg:.2e} | {imp_str} | {cl:.2e} | {rl:.2e} |")
    
    with open(model_dir / "rq1_table.md", "w") as f:
        f.write("\n".join(table_lines))
        
    # 2. Generate Horizontal Bar Charts (Image 1 style)
    # We will plot all sizes to give a complete view of the progression
    s_sub = sizes
    cg_sub = [g * 1e5 for g in cte_global]
    rg_sub = [g * 1e5 for g in rand_global]
    cl_sub = [l * 1e5 for l in cte_local]
    rl_sub = [l * 1e5 for l in rand_local]
    
    y = np.arange(len(s_sub))
    height = 0.35
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1.barh(y + height/2, cg_sub, height, label='CTE', color='#c44e52')
    ax1.barh(y - height/2, rg_sub, height, label='random', color='#4c72b0')
    ax1.set_yticks(y)
    ax1.set_yticklabels([f'N={s}' for s in s_sub])
    ax1.invert_yaxis()  # labels read top-to-bottom
    ax1.set_xlabel('Global MAE ($\\times 10^{-5}$)')
    ax1.set_title('A) Global Feature Importance', loc='left', fontweight='bold')
    ax1.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax1.legend(loc='lower right')
    
    ax2.barh(y + height/2, cl_sub, height, label='CTE', color='#c44e52')
    ax2.barh(y - height/2, rl_sub, height, label='random', color='#4c72b0')
    ax2.set_yticks(y)
    ax2.set_yticklabels([f'N={s}' for s in s_sub])
    ax2.invert_yaxis()
    ax2.set_xlabel('Local MAE ($\\times 10^{-5}$)')
    ax2.set_title('B) Per-Sample Explanation', loc='left', fontweight='bold')
    ax2.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax2.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(model_dir / 'rq1_plot.png', dpi=300)
    plt.close()

def plot_rq2(model_dir, data):
    windows = [w['window'] for w in data['windows']]
    
    # --- Plot 1: Global MAE over time (Fixed X-axis) ---
    mae_cte = []
    mae_rand = []
    for w in data['windows']:
        t = np.array(w['truth_importance'])
        c = np.array(w['cte_importance'])
        r = np.array(w['rand_importance'])
        mae_cte.append(np.mean(np.abs(t - c)))
        mae_rand.append(np.mean(np.abs(t - r)))
        
    plt.figure(figsize=(8, 5))
    plt.plot(windows, mae_cte, marker='o', label='CTE (Size 128)', color='#c44e52', linewidth=2)
    plt.plot(windows, mae_rand, marker='s', label='Random (Size 128)', color='#4c72b0', linewidth=2)
    plt.title(f'{model_dir.name}: Temporal Drift (Global MAE over Windows)')
    plt.xlabel('Time Window')
    plt.ylabel('Global MAE')
    plt.xticks(windows) # Forces integer x-axis ticks
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / 'rq2_mae_plot.png', dpi=300)
    plt.close()

    # --- Plot 2: Feature Tracking over Time (Image 3 style) ---
    feature_names = data['feature_names']
    # Find top 5 features in window 1 truth
    w1_truth = data['windows'][0]['truth_importance']
    top_5_idx = np.argsort(w1_truth)[-5:][::-1]
    top_5_names = [feature_names[i] for i in top_5_idx]
    
    # Extract series for these 5 features
    truth_series = {name: [] for name in top_5_names}
    cte_series = {name: [] for name in top_5_names}
    rand_series = {name: [] for name in top_5_names}
    
    for w in data['windows']:
        for i, name in zip(top_5_idx, top_5_names):
            truth_series[name].append(w['truth_importance'][i])
            cte_series[name].append(w['cte_importance'][i])
            rand_series[name].append(w['rand_importance'][i])
            
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    colors = ['#c44e52', '#4c72b0', '#eebd52', '#55a868', '#4c4c4c']
    markers = ['o', 'o', 'o', 'o', 'o']
    
    for ax, title, series_dict, letter in zip(axes, 
                                      ['Ground Truth', 'CTE (N=128)', 'Random (N=128)'], 
                                      [truth_series, cte_series, rand_series],
                                      ['A)', 'B)', 'C)']):
        for color, marker, name in zip(colors, markers, top_5_names):
            ax.plot(windows, series_dict[name], marker=marker, color=color, linewidth=2, label=name)
        
        ax.set_title(title)
        ax.set_title(letter, loc='left', fontweight='bold')
        ax.set_xlabel('Time Window')
        ax.set_xticks(windows)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        if letter == 'A)':
            ax.set_ylabel('Mean |SHAP|')
            
    # Unified legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(model_dir / 'rq2_features_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_rq3(model_dir, data):
    feature_names = data['feature_names']
    train_imp = np.array(data['train']['truth_importance'])
    val_imp = np.array(data['val']['truth_importance'])
    
    # Select top 10 features from train
    top_10_idx = np.argsort(train_imp)[-10:][::-1]
    
    names = [feature_names[i] for i in top_10_idx]
    t_vals = [train_imp[i] for i in top_10_idx]
    v_vals = [val_imp[i] for i in top_10_idx]
    
    y = np.arange(len(names))
    height = 0.35
    
    plt.figure(figsize=(10, 7))
    # Note: image 4 has train (red) on top, validation (blue) on bottom for each feature
    # So we plot train at y - height/2 and val at y + height/2, then invert y-axis
    bars_train = plt.barh(y - height/2, t_vals, height, label='train', color='#c44e52')
    bars_val = plt.barh(y + height/2, v_vals, height, label='validation', color='#4c72b0')
    
    plt.yticks(y, names)
    plt.gca().invert_yaxis()
    plt.xlabel('Mean |SHAP|')
    plt.title('A) Train vs Validation Importance', loc='left', fontweight='bold')
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    # Add multiplier annotations
    for i in range(len(y)):
        tv = t_vals[i]
        vv = v_vals[i]
        if tv > 0:
            multiplier = vv / tv
            # place text at the end of the longer bar
            max_val = max(tv, vv)
            plt.text(max_val + max(t_vals)*0.01, y[i] + height/4, f'{multiplier:.1f}$\\times$', va='center', fontsize=9)
            
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(model_dir / 'rq3_plot.png', dpi=300)
    plt.close()

def plot_rq4(model_dir, data):
    n_batch = data['n_explain_per_batch']
    
    # We will use all sizes for the Break-even plot to answer the user's request
    sizes = [exp['size'] for exp in data['experiments']]
    exps = data['experiments']
    
    # We might have negative deltas due to microsecond jitter in TreeExplainer
    # We will clamp delta to a tiny positive number so it mathematically breaks even at a huge number,
    # or we just explicitly skip it but annotate it properly.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # A) Cumulative Savings
    # To avoid extreme clutter in the line plot, we pick 5 representative sizes
    line_sizes = [8, 16, 64, 256, 1024]
    line_exps = [e for e in exps if e['size'] in line_sizes]
    if not line_exps: line_exps = exps
    
    colors = plt.cm.tab10.colors
    
    # Calculate a sensible max X for the line plot based on models that break even in a reasonable time
    reasonable_n_stars = [e['n_star_batches'] for e in line_exps if e['n_star_batches'] is not None and e['n_star_batches'] > 0]
    if reasonable_n_stars:
        max_batches = max(reasonable_n_stars) * 1.5
    else:
        max_batches = 1000 # default
        
    x_batches = np.linspace(0, max_batches, 100)
    
    for i, exp in enumerate(line_exps):
        size = exp['size']
        build = exp['build_time']
        delta = exp['delta']
        n_star = exp['n_star_batches']
        
        c = colors[i % len(colors)]
        
        if delta <= 0 or n_star is None:
            # It never breaks even due to evaluation being slower than or equal to ground truth
            y_savings = (delta * x_batches) - build
            label = f"bg={size} (no speedup)"
            ax1.plot(x_batches, y_savings, label=label, color=c, linewidth=2, linestyle=':')
        else:
            y_savings = (delta * x_batches) - build
            label = f"bg={size} (n*={int(n_star)})"
            ax1.plot(x_batches, y_savings, label=label, color=c, linewidth=2)
            if n_star <= max_batches:
                ax1.axvline(x=n_star, color=c, linestyle=':', alpha=0.7)
        
    ax1.axhline(y=0, color='k', linestyle='--', label='break-even')
    ax1.set_title('A) Cumulative Savings', loc='left', fontweight='bold')
    ax1.set_xlabel(f'Number of explanation batches ({n_batch} samples/batch)')
    ax1.set_ylabel('Cumulative time saved by CTE vs Truth [s]')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # B) Break-even Point (Bar Chart)
    # We plot all sizes for the bar chart
    bg_labels = []
    n_stars = []
    n_pts = []
    bar_colors = []
    
    for i, exp in enumerate(exps):
        if exp['delta'] > 0 and exp['n_star_batches'] is not None:
            bg_labels.append(f"bg={exp['size']}")
            n_stars.append(exp['n_star_batches'])
            n_pts.append(exp['n_star_samples'])
            bar_colors.append(colors[i % len(colors)])
            
    y = np.arange(len(n_stars))
    if len(y) > 0:
        bars = ax2.barh(y, n_stars, height=0.6, color=bar_colors)
        
        ax2.set_yticks(y)
        ax2.set_yticklabels(bg_labels)
        ax2.invert_yaxis()
        ax2.set_title('B) Break-even Point (n*)', loc='left', fontweight='bold')
        ax2.set_xlabel('Break-even batches (n*)')
        ax2.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Determine the maximum x-value to set limits properly
        max_n_star = max(n_stars)
        ax2.set_xlim(0, max_n_star * 1.3) # Add 30% space for text annotations
        
        for i, bar in enumerate(bars):
            width = bar.get_width()
            pts = int(n_pts[i])
            ax2.text(width + max_n_star*0.02, bar.get_y() + bar.get_height()/2, 
                     f'{int(width)} ({pts:,} pts)', va='center', fontsize=10)
    else:
        ax2.text(0.5, 0.5, "No CTE sizes reached break-even\n(Eval time jitter > savings)", ha='center', va='center', transform=ax2.transAxes)
                 
    plt.suptitle(f'{model_dir.name}: CTE pays off when explanations are reused', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(model_dir / 'rq4_plot.png', dpi=300)
    plt.close()

def main():
    base_dir = Path(__file__).parent.parent.parent / "results" / "fallback_plan"
    
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
                
        if (model_dir / "rq4.json").exists():
            with open(model_dir / "rq4.json") as f:
                plot_rq4(model_dir, json.load(f))

if __name__ == "__main__":
    main()
    print("All precise visualizations generated successfully!")
