import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load_all_rq1_data(base_dir):
    tree_models = ['xgb_tree', 'rf_tree', 'lgb_tree', 'catboost_tree']
    linear_models = ['logreg_linear', 'ridge_linear', 'lasso_linear', 'svc_linear']
    
    tree_data = []
    linear_data = []
    
    for model_name in tree_models:
        rq1_path = base_dir / model_name / 'rq1.json'
        if rq1_path.exists():
            with open(rq1_path) as f:
                tree_data.append(json.load(f)['experiments'])
                
    for model_name in linear_models:
        rq1_path = base_dir / model_name / 'rq1.json'
        if rq1_path.exists():
            with open(rq1_path) as f:
                linear_data.append(json.load(f)['experiments'])
                
    return tree_data, linear_data

def aggregate_data(family_data):
    # family_data is a list of experiment lists: [ [exp1, exp2...], [exp1, exp2...] ]
    # We assume all models have the same sizes in the same order
    if not family_data: return [], [], [], [], []
    
    n_models = len(family_data)
    n_sizes = len(family_data[0])
    
    sizes = [family_data[0][i]['size'] for i in range(n_sizes)]
    
    agg_cte_global = np.zeros(n_sizes)
    agg_rand_global = np.zeros(n_sizes)
    agg_cte_local = np.zeros(n_sizes)
    agg_rand_local = np.zeros(n_sizes)
    
    for model_exps in family_data:
        for i, exp in enumerate(model_exps):
            agg_cte_global[i] += exp['cte']['mae_global']
            agg_rand_global[i] += exp['random']['mae_global']
            agg_cte_local[i] += exp['cte']['mae_local']
            agg_rand_local[i] += exp['random']['mae_local']
            
    # Average
    agg_cte_global /= n_models
    agg_rand_global /= n_models
    agg_cte_local /= n_models
    agg_rand_local /= n_models
    
    return sizes, agg_cte_global, agg_rand_global, agg_cte_local, agg_rand_local

def plot_aggregated_rq4(base_dir):
    tree_data, linear_data = load_all_rq1_data(base_dir)
    
    t_sizes, t_cg, t_rg, t_cl, t_rl = aggregate_data(tree_data)
    l_sizes, l_cg, l_rg, l_cl, l_rl = aggregate_data(linear_data)
    
    if not t_sizes and not l_sizes:
        print("No data found to aggregate.")
        return
        
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Custom vibrant colors
    c_cte = '#c44e52'
    c_rand = '#4c72b0'
    
    def plot_panel(ax, sizes, cte_vals, rand_vals, title, ylabel, show_legend=False):
        ax.plot(sizes, cte_vals, marker='o', color=c_cte, linewidth=2, label='CTE')
        ax.plot(sizes, rand_vals, marker='s', color=c_rand, linewidth=2, label='Random')
        ax.set_title(title, loc='left', fontweight='bold')
        ax.set_xlabel('Background Size (N)')
        ax.set_ylabel(ylabel)
        ax.set_xscale('log', base=2)
        ax.set_xticks(sizes)
        ax.set_xticklabels(sizes)
        ax.grid(True, alpha=0.3)
        if show_legend:
            ax.legend()
            
    # Top Row: Trees
    plot_panel(axes[0, 0], t_sizes, t_cg, t_rg, 'A) Tree Models: Global Feature Importance', 'Global MAE', show_legend=True)
    plot_panel(axes[0, 1], t_sizes, t_cl, t_rl, 'B) Tree Models: Per-Sample Explanation', 'Local MAE')
    
    # Bottom Row: Linears
    plot_panel(axes[1, 0], l_sizes, l_cg, l_rg, 'C) Linear Models: Global Feature Importance', 'Global MAE')
    plot_panel(axes[1, 1], l_sizes, l_cl, l_rl, 'D) Linear Models: Per-Sample Explanation', 'Local MAE')
    
    plt.suptitle('RQ4: CTE Architecture Agnosticism (Averaged Across Model Families)', fontsize=18, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    out_file = base_dir / 'rq4_aggregated.png'
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Aggregated RQ4 plot saved to {out_file}")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent.parent / "results" / "fallback_plan"
    plot_aggregated_rq4(base_dir)
