import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, confusion_matrix
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from data.data_loader import load_dataset
from experiments.main_pipeline.common import PAIRS

def generate_performance():
    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache_full'
    data = load_dataset(data_dir=Path(__file__).parent.parent.parent / 'data' / 'ecom-offers')
    
    # Use test set if available, else validation set
    X_eval = data['X_test'] if data['X_test'] is not None else data['X_val']
    y_eval = data['y_test'] if data['y_test'] is not None else data['y_val']
    set_name = "Test" if data['X_test'] is not None else "Validation"
    
    print(f"Evaluating models on {set_name} set ({len(X_eval)} samples)...")
    
    results = []
    
    pretty_names = {
        'xgb_tree': 'XGBoost', 'rf_tree': 'Random Forest',
        'lgb_tree': 'LightGBM', 'catboost_tree': 'CatBoost',
        'logreg_linear': 'LogReg', 'ridge_linear': 'Ridge',
        'lasso_linear': 'Lasso', 'svc_linear': 'SVC',
    }
    
    for pair in PAIRS:
        model_path = cache_dir / pair['model_file']
        if not model_path.exists():
            print(f"Skipping {pair['name']} - model not found")
            continue
            
        model = joblib.load(model_path)
        
        # Predict
        if pair['family'] == 'linear':
            # Linear models might be pipelines
            y_pred = model.predict(X_eval)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_eval)[:, 1]
            else:
                y_prob = model.decision_function(X_eval)
        else:
            y_pred = model.predict(X_eval)
            y_prob = model.predict_proba(X_eval)[:, 1]
            
        # Binary rounding for y_pred if necessary
        y_pred = (y_pred > 0.5).astype(int)
        
        auc = roc_auc_score(y_eval, y_prob)
        acc = accuracy_score(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred)
        
        results.append({
            'Model': pretty_names.get(pair['name'], pair['name']),
            'Family': 'Tree' if pair['family'] == 'tree' else 'Linear',
            'ROC-AUC': auc,
            'Accuracy': acc,
            'F1-Score': f1
        })
    
    # Create LaTeX Table
    df = pd.DataFrame(results)
    
    latex_table = "\\begin{table}[h!]\n\\centering\n"
    latex_table += "\\caption{Predictive Performance of Evaluated Models on the " + set_name + " Set}\n"
    latex_table += "\\label{tab:model_performance}\n"
    latex_table += "\\begin{tabular}{llccc}\n"
    latex_table += "\\toprule\n"
    latex_table += "\\textbf{Model} & \\textbf{Family} & \\textbf{ROC-AUC} & \\textbf{Accuracy} & \\textbf{F1-Score} \\\\\n"
    latex_table += "\\midrule\n"
    
    for _, row in df.iterrows():
        latex_table += f"{row['Model']} & {row['Family']} & {row['ROC-AUC']:.4f} & {row['Accuracy']:.4f} & {row['F1-Score']:.4f} \\\\\n"
        
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}\n"
    
    print("\n" + latex_table)
    
    out_path = Path(__file__).parent.parent.parent / 'results' / 'main_pipeline' / 'performance_table.tex'
    with open(out_path, 'w') as f:
        f.write(latex_table)
    print(f"\nSaved LaTeX table to {out_path}")

if __name__ == "__main__":
    generate_performance()
