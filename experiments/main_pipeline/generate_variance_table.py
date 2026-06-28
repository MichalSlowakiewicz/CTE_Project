import json
from pathlib import Path
import pandas as pd

def generate_variance():
    base_dir = Path(__file__).parent.parent.parent / 'results' / 'main_pipeline'
    
    models = [d.name for d in base_dir.iterdir() if d.is_dir()]
    
    results = []
    
    pretty_names = {
        'xgb_tree': 'XGBoost', 'rf_tree': 'Random Forest',
        'lgb_tree': 'LightGBM', 'catboost_tree': 'CatBoost',
        'logreg_linear': 'LogReg', 'ridge_linear': 'Ridge',
        'lasso_linear': 'Lasso', 'svc_linear': 'SVC',
    }
    
    for m in sorted(models):
        rq1_path = base_dir / m / 'rq1.json'
        if not rq1_path.exists():
            continue
            
        with open(rq1_path, 'r') as f:
            data = json.load(f)
            
        # Find N=128
        for exp in data['experiments']:
            if exp['size'] == 128:
                if 'mae_global_std' in exp['cte']:
                    cte_std = exp['cte']['mae_global_std']
                    rand_std = exp['random']['mae_global_std']
                    
                    results.append({
                        'Model': pretty_names.get(m, m),
                        'Family': 'Tree' if 'tree' in m else 'Linear',
                        'CTE Std': cte_std,
                        'Rand Std': rand_std,
                        'Reduction': ((rand_std - cte_std) / rand_std) * 100 if rand_std > 0 else 0
                    })
                break
                
    # Create LaTeX Table
    df = pd.DataFrame(results)
    
    latex_table = "\\begin{table}[h!]\n\\centering\n"
    latex_table += "\\caption{Global MAE Standard Deviation at $N=128$ (30 independent seeds)}\n"
    latex_table += "\\label{tab:variance_reduction}\n"
    latex_table += "\\begin{tabular}{llccc}\n"
    latex_table += "\\toprule\n"
    latex_table += "\\textbf{Model} & \\textbf{Family} & \\textbf{Random Std ($\\sigma$)} & \\textbf{CTE Std ($\\sigma$)} & \\textbf{Variance Reduction} \\\\\n"
    latex_table += "\\midrule\n"
    
    for _, row in df.iterrows():
        latex_table += f"{row['Model']} & {row['Family']} & {row['Rand Std']:.2e} & {row['CTE Std']:.2e} & \\textbf{{{row['Reduction']:.1f}\\%}} \\\\\n"
        
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}\n"
    
    print("\n" + latex_table)
    
    out_path = base_dir / 'variance_table.tex'
    with open(out_path, 'w') as f:
        f.write(latex_table)
    print(f"\nSaved LaTeX table to {out_path}")

if __name__ == "__main__":
    generate_variance()
