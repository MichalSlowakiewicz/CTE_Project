import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def plot_rq1(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
        
    experiments = data['experiments']
    sizes = [exp['size'] for exp in experiments]
    
    cte_mae = [exp['cte']['mae_global'] for exp in experiments]
    rand_mae = [exp['random']['mae_global'] for exp in experiments]
    
    cte_speedup = [exp['cte']['speedup_vs_truth'] for exp in experiments]
    rand_speedup = [exp['random']['speedup_vs_truth'] for exp in experiments]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Fidelity vs Size
    ax1.plot(sizes, cte_mae, marker='o', label='CTE (Kernel Thinning)', color='darkblue')
    ax1.plot(sizes, rand_mae, marker='s', label='Random Background', color='darkorange')
    ax1.set_xlabel('Background Size ($N_{bg}$)')
    ax1.set_ylabel('Mean Absolute Error (MAE)')
    ax1.set_title('Explanation Error (Global Importance, Lower is Better)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup vs Size
    ax2.plot(sizes, cte_speedup, marker='o', label='CTE Speedup', color='darkblue')
    ax2.plot(sizes, rand_speedup, marker='s', label='Random Speedup', color='darkorange')
    ax2.set_xlabel('Background Size ($N_{bg}$)')
    ax2.set_ylabel('Speedup Ratio vs Ground Truth')
    ax2.set_title('Computational Efficiency')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(results_file).parent / 'rq1_efficiency.png', dpi=300)
    plt.close()

def plot_rq2(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
        
    windows = data['windows']
    feature_names = data['feature_names']
    
    # Get top 5 features from first window GT to track
    gt_first = np.array(windows[0]['truth_importance'])
    top_indices = np.argsort(gt_first)[-5:][::-1]
    
    # Extract temporal data for these features
    n_windows = len(windows)
    time_axis = np.arange(1, n_windows + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    
    for idx, ax, method in zip([0, 1, 2], axes, ['truth_importance', 'cte_importance', 'rand_importance']):
        for feat_idx in top_indices:
            feat_name = feature_names[feat_idx]
            vals = [w[method][feat_idx] for w in windows]
            ax.plot(time_axis, vals, marker='o', label=feat_name)
            
        ax.set_xlabel('Time Window')
        title = "Ground Truth" if method == 'truth_importance' else ("CTE" if method == 'cte_importance' else "Random")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(time_axis)
        
    axes[0].set_ylabel('Mean Absolute SHAP Value')
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(Path(results_file).parent / 'rq2_drift.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def plot_rq3(results_file):
    with open(results_file, 'r') as f:
        data = json.load(f)
        
    feature_names = data['feature_names']
    
    # Get top 10 features overall from train
    train_gt = np.array(data['train']['truth_importance'])
    top_indices = np.argsort(train_gt)[-10:][::-1]
    
    labels = [feature_names[i] for i in top_indices]
    
    train_vals = [data['train']['truth_importance'][i] for i in top_indices]
    val_vals = [data['val']['truth_importance'][i] for i in top_indices]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, train_vals, width, label='Training Data', color='teal')
    ax.bar(x + width/2, val_vals, width, label='Validation Data', color='coral')
    
    ax.set_ylabel('Mean Absolute SHAP Value')
    ax.set_title('Feature Importance Discrepancy: Train vs Validation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(Path(results_file).parent / 'rq3_train_val.png', dpi=300)
    plt.close()

def plot_breakeven(results_file):
    """
    Break-even analysis using KernelExplainer timings from rq4_breakeven.json.

    Shows cumulative time SAVED by using CTE instead of Random:
        savings(n) = n * delta - build_time

    Starts negative (you're in debt from building the coreset),
    crosses zero at n* (break-even), then goes positive (CTE pays off).
    This makes the break-even point visually obvious.
    """
    with open(results_file) as f:
        data = json.load(f)

    experiments = data['experiments']
    n_explain   = data['n_explain_per_batch']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # x-axis: go to 2× the largest n* so all break-even points are visible
    max_n_star = max(
        exp['build_time'] / (exp['t_rand_shap'] - exp['t_cte_shap'])
        for exp in experiments
        if (exp['t_rand_shap'] - exp['t_cte_shap']) > 0
    )
    max_batches = int(max_n_star * 2)

    fig, ax = plt.subplots(figsize=(10, 6))

    for exp, color in zip(experiments, colors):
        size       = exp['size']
        build_time = exp['build_time']
        delta      = exp['t_rand_shap'] - exp['t_cte_shap']

        if delta <= 0:
            continue

        n_batches = np.arange(0, max_batches + 1)
        savings   = n_batches * delta - build_time  # negative → zero → positive

        n_star = build_time / delta

        ax.plot(n_batches, savings, color=color, linewidth=2,
                label=f'bg={size}  (n*={n_star:.0f} batches = {n_star*n_explain:.0f} samples)')
        ax.axvline(n_star, color=color, linestyle=':', alpha=0.5)

    ax.axhline(0, color='black', linewidth=1.2, linestyle='--', label='Break-even (savings = 0)')
    ax.fill_between(np.arange(0, max_batches + 1), 0,
                    alpha=0.04, color='green')

    ax.set_xlabel(f'Number of explanation batches ({n_explain} samples/batch)')
    ax.set_ylabel('Cumulative time saved by CTE vs Random (s)')
    ax.set_title('CTE Break-even Analysis (KernelExplainer)\n'
                 'Below zero = CTE still in debt from build cost  |  Above zero = CTE pays off')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = Path(results_file).parent / 'rq4_breakeven.png'
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Break-even plot saved to {out_path}")


if __name__ == "__main__":
    results_dir = Path(__file__).parent.parent / 'results'
    if (results_dir / 'rq1.json').exists(): plot_rq1(results_dir / 'rq1.json')
    if (results_dir / 'rq2.json').exists(): plot_rq2(results_dir / 'rq2.json')
    if (results_dir / 'rq3.json').exists(): plot_rq3(results_dir / 'rq3.json')
    if (results_dir / 'rq4_breakeven.json').exists():
        plot_breakeven(results_dir / 'rq4_breakeven.json')
    print("Plots generated in results directory.")
