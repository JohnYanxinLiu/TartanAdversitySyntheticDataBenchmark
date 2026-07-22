import wandb
import pandas as pd
import re
import os

# --- Configuration ---
# Weights & Biases entity/project come from the environment (see ../.env.example).
ENTITY = os.getenv("WANDB_ENTITY")
PROJECT = os.getenv("WANDB_PROJECT")
if not PROJECT:
    raise SystemExit(
        "Set WANDB_ENTITY and WANDB_PROJECT before running "
        "(e.g. `source .env`; see .env.example)."
    )
OUTPUT_DIR = "wandb_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

api = wandb.Api()

models = ["resnet", "convnext", "yolo"]
aug_types = ["gemini", "automold"]

# Canonical per-weather metrics logged by all three models. DAWN's haze/mist/fog
# are accumulated into "foggy" upstream, so only clear/rainy/foggy/snowy appear.
metrics_of_interest = ["val/mAP", "val/clear/mAP", "val/rainy/mAP",
                       "val/foggy/mAP", "val/snowy/mAP",
                       "foggy_zurich/mAP",
                       "dawn/mAP", "dawn/rainy/mAP", "dawn/foggy/mAP", "dawn/snowy/mAP",
                       "acdc_train/mAP",
                       "acdc_train/rainy/mAP", "acdc_train/foggy/mAP", "acdc_train/snowy/mAP",
                       "acdc_val/mAP",
                       "acdc_val/rainy/mAP", "acdc_val/foggy/mAP", "acdc_val/snowy/mAP"]

# Runs recorded before weather labels were canonicalized logged raw dataset
# vocabulary (e.g. "acdc_train/fog/mAP", "dawn/rain/mAP"). Fall back to these
# when the canonical column is absent so old runs still extract fully.
# DAWN's legacy haze/mist slices are intentionally not folded in here: the
# canonical "foggy" metric is computed over the combined image set, which is
# not recoverable from separate per-slice maxima.
legacy_aliases = {
    "dawn/rainy/mAP": ["dawn/rain/mAP"],
    "dawn/foggy/mAP": ["dawn/fog/mAP"],
    "dawn/snowy/mAP": ["dawn/snow/mAP"],
    "acdc_train/rainy/mAP": ["acdc_train/rain/mAP"],
    "acdc_train/foggy/mAP": ["acdc_train/fog/mAP"],
    "acdc_train/snowy/mAP": ["acdc_train/snow/mAP"],
    "acdc_val/rainy/mAP": ["acdc_val/rain/mAP"],
    "acdc_val/foggy/mAP": ["acdc_val/fog/mAP"],
    "acdc_val/snowy/mAP": ["acdc_val/snow/mAP"],
}


def parse_run_name(name):
    # The suffix (_noaug) is now optional: (_noaug)?
    pattern = r"^(" + "|".join(models) + r")_([ra])(\d{2})_(" + "|".join(aug_types) + r")(?:_noaug)?$"
    match = re.match(pattern, name)
    if match:
        return {
            "model": match.group(1),
            "mix_type": match.group(2),
            "mix_value": match.group(3),
            "aug_type": match.group(4)
        }
    else:
        print(f"Warning: Run name '{name}' did not match expected pattern.")
    return None

all_rows = []
runs = api.runs(f"{ENTITY}/{PROJECT}")

for run in runs:
    metadata = parse_run_name(run.name)
    
    if not metadata:
        continue

    # Pull the (sampled) full history once; evaluation is logged sparsely so
    # 2000 samples covers every logged epoch. This avoids run.history_keys,
    # which no longer exists in recent wandb API versions.
    hist = run.history(samples=2000)

    if hist.empty:
        continue

    # For each canonical metric take the max over history, falling back to
    # the legacy (pre-canonicalization) column name for old runs.
    best_metrics = {}
    for metric in metrics_of_interest:
        for column in [metric] + legacy_aliases.get(metric, []):
            if column in hist.columns:
                best = hist[column].max()
                if pd.notna(best):
                    best_metrics[metric] = best
                    break

    if best_metrics:
        all_rows.append({
            **metadata,
            **best_metrics,
            "run_id": run.id,
            "run_name": run.name
        })

# --- Grouping and Saving ---
if all_rows:
    df_main = pd.DataFrame(all_rows)
    grouped = df_main.groupby(['model', 'mix_type'])
    # import pdb; pdb.set_trace()

    for (model_name, mix_type), group_df in grouped:
        group_df = group_df.sort_values(by=["mix_value", "aug_type"])
        filename = f"{model_name}_{mix_type}_results.csv"
        group_df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
        print(f"Saved {filename}")
else:
    print("No runs matched the naming criteria.")