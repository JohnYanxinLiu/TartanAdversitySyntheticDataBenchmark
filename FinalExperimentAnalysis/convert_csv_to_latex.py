import pandas as pd
import numpy as np

def generate_latex_rows(repl_csv, add_csv, model_label):
    # Load data
    df_repl = pd.read_csv(repl_csv, sep=None, engine='python')
    df_add = pd.read_csv(add_csv, sep=None, engine='python')

    # Define the columns for each group in the order they appear in the LaTeX header
    bdd_cols = ['val/mAP', 'val/snowy/mAP', 'val/rainy/mAP', 'val/foggy/mAP', 'val/clear/mAP']
    ood_cols = ['dawn/mAP', 'dawn/rainy/mAP', 'dawn/snowy/mAP', 'dawn/foggy/mAP', 'foggy_zurich/mAP']
    acdc_cols = ['acdc_val/mAP', 'acdc_val/snowy/mAP', 'acdc_val/rainy/mAP', 'acdc_val/foggy/mAP']

    def get_val(df, mix_val, aug_type, col):
        subset = df[(df['mix_value'] == mix_val) & (df['aug_type'].str.lower() == aug_type.lower())]
        if subset.empty:
            return "--"
        val = subset[col].values[0]
        return f"{val * 100:.1f}"

    def build_row(df, mix_val, type_label, is_first_for_type=True):
        label = type_label if is_first_for_type else ""
        # Percent is shown only on the first row of a group or as specified
        perc = str(int(mix_val))
        
        parts = [label, perc]
        
        # BDD Group
        for col in bdd_cols:
            parts.append(get_val(df, mix_val, 'gemini', col))
            parts.append(get_val(df, mix_val, 'automold', col))
        
        parts.append("") # Spacer
        
        # OOD Group
        for col in ood_cols:
            parts.append(get_val(df, mix_val, 'gemini', col))
            parts.append(get_val(df, mix_val, 'automold', col))
            
        parts.append("") # Spacer
        
        # ACDC Group
        for col in acdc_cols:
            parts.append(get_val(df, mix_val, 'gemini', col))
            parts.append(get_val(df, mix_val, 'automold', col))
            
        return " & ".join(parts) + " \\\\"

    output = []
    output.append(f"\\rowcolor{{gray!15}} \\multicolumn{{32}}{{l}}{{\\textbf{{{model_label}}}}} \\\\")

    # 1. Base Row (mix_value 0 from Replacement file)
    output.append(build_row(df_repl, 0, "Base"))
    output.append("\\midrule")

    # 2. Replacement Rows
    repl_vals = sorted([v for v in df_repl['mix_value'].unique() if v > 0])
    for i, v in enumerate(repl_vals):
        output.append(build_row(df_repl, v, "Repl.", is_first_for_type=(i==0)))
    
    output.append("\\midrule")

    # 3. Additive Rows
    add_vals = sorted([v for v in df_add['mix_value'].unique() if v > 0])
    for i, v in enumerate(add_vals):
        output.append(build_row(df_add, v, "Add.", is_first_for_type=(i==0)))

    return "\n".join(output)

# --- EXECUTION ---
# Update these filenames to match yours
models = [
    {"repl": "resnet_r_results.csv", "add": "resnet_a_results.csv", "label": "ResNet-50"},
    {"repl": "convnext_r_results.csv", "add": "convnext_a_results.csv", "label": "ConvNext-Tiny"},
    {"repl": "yolo_r_results.csv", "add": "yolo_a_results.csv", "label": "YOLOv8"}
]

import os
csv_dir = "wandb_results"

for model in models:
    try:
        print(f"\n% --- {model['label']} ---")
        print(generate_latex_rows(os.path.join(csv_dir, model['repl']), os.path.join(csv_dir, model['add']), model['label']))
        print("\\midrule")
    except Exception as e:
        print(f"% Error processing {model['label']}: {e}")