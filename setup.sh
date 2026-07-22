#!/usr/bin/env bash
#
# One-shot environment + dataset setup for the TartanAdversity benchmark.
#
#   ./setup.sh                       # create .venv, install deps, build all datasets
#   ./setup.sh --max-train 800       # smaller clear-weather training subset (quick run)
#   ./setup.sh --skip-env            # skip env creation/install (use the active env)
#   PYTHON=python3.11 ./setup.sh     # choose the interpreter used to create .venv
#
# ACDC is registration-gated and cannot be downloaded automatically. This script
# checks for it up front and tells you how to obtain it, then proceeds with every
# other dataset regardless; ACDC is preprocessed at the end only if its archives
# are present. The script is idempotent — already-built datasets are skipped, so
# it is safe to re-run (e.g. after dropping in the ACDC archives).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MAX_TRAIN=28800          # clear-weather training images (0 = all; the study used 28800)
SKIP_ENV=false
PYTHON="${PYTHON:-python3}"

while [ $# -gt 0 ]; do
  case "$1" in
    --max-train) MAX_TRAIN="$2"; shift 2 ;;
    --skip-env)  SKIP_ENV=true; shift ;;
    -h|--help)   awk 'NR==1{next} /^#/{sub(/^# ?/,""); print; next} {exit}' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

ACDC_DIR="dataset_utils/acdc_dataset_zip_files"

# --- status table on exit (prints even if a step fails) ------------------------
summary() {
  echo
  echo "======================= dataset status ======================="
  _row() { if [ -e "$1" ]; then printf "  [ready]   %s\n" "$2"; else printf "  [MISSING] %s\n" "$2"; fi; }
  _row Data/ClearRealSubset/combined_dataset_coco.json "BDD100K ClearRealSubset (real train)"
  _row Data/bdd100k_val/bdd100k_val_coco.json          "BDD100K val (in-distribution)"
  _row Data/GeminiAugmented/GeminiFogCoco.json         "TartanAdversity synthetic (Gemini + Automold)"
  _row Data/DAWN/dawn_coco.json                        "DAWN"
  _row Data/FoggyZurich/foggy_zurich_coco.json         "Foggy Zurich"
  _row Data/ACDC/acdc_train_coco.json                  "ACDC (train + val)"
  echo "=============================================================="
}
trap summary EXIT

# --- 1. environment ------------------------------------------------------------
if [ "$SKIP_ENV" = false ]; then
  if [ -z "${VIRTUAL_ENV:-}" ] && [ -z "${CONDA_PREFIX:-}" ]; then
    if [ ! -d .venv ]; then
      echo "[env] creating virtualenv at .venv (interpreter: $PYTHON)"
      "$PYTHON" -m venv .venv
    fi
    set +u; source .venv/bin/activate; set -u
    echo "[env] activated .venv"
  else
    echo "[env] using already-active environment (${VIRTUAL_ENV:-$CONDA_PREFIX})"
  fi
  echo "[env] installing requirements (this can take a while: torch, ultralytics, ...)"
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet -r requirements.txt
else
  echo "[env] --skip-env: using current 'python' without installing anything"
fi

# Xet transfer can stall on some networks; plain HTTPS is more reliable here.
export HF_HUB_DISABLE_XET=1

# --- 2. ACDC check (first, because it is the only manual dataset) --------------
acdc_ready=false
if [ -f "$ACDC_DIR/rgb_anon_trainvaltest.zip" ]; then
  if [ ! -d "$ACDC_DIR/gt_detection" ] && [ -f "$ACDC_DIR/gt_detection_trainval.zip" ]; then
    echo "[acdc] extracting gt_detection_trainval.zip ..."
    python -c "import zipfile; zipfile.ZipFile('$ACDC_DIR/gt_detection_trainval.zip').extractall('$ACDC_DIR')"
  fi
  if [ -d "$ACDC_DIR/gt_detection" ]; then
    acdc_ready=true
    echo "[acdc] archives found — will preprocess after the other datasets."
  fi
fi
if [ "$acdc_ready" = false ] && [ ! -f Data/ACDC/acdc_train_coco.json ]; then
  cat <<EOF
[acdc] ACDC archives not found. ACDC is registration-gated and cannot be
       downloaded automatically. To include it:
         1. Register and download at https://acdc.vision.ee.ethz.ch/
         2. Put   rgb_anon_trainvaltest.zip   in $ACDC_DIR/
         3. Put   gt_detection_trainval.zip   in $ACDC_DIR/  (this script will unzip it)
       Then re-run ./setup.sh. Continuing with the other datasets for now.
EOF
fi

# --- 3. auto-downloadable datasets --------------------------------------------
# BDD100K (real): download + COCO conversion, then the clear-weather train subset.
if [ ! -f Data/BDD100K/bdd100k_train_coco.json ]; then
  echo "[bdd100k] downloading + converting to COCO ..."
  python dataset_utils/download_bdd100k.py
else
  echo "[bdd100k] already downloaded — skipping."
fi
if [ ! -f Data/ClearRealSubset/combined_dataset_coco.json ]; then
  echo "[bdd100k] building ClearRealSubset (max_train=$MAX_TRAIN) + bdd100k_val ..."
  python dataset_utils/build_clear_real_subset.py --max_train "$MAX_TRAIN"
else
  echo "[bdd100k] ClearRealSubset already built — skipping."
fi

# TartanAdversity synthetic (Gemini + Automold), incl. ranking CSVs from the Hub.
if [ ! -f Data/GeminiAugmented/GeminiFogCoco.json ]; then
  echo "[synthetic] downloading TartanAdversity from Hugging Face ..."
  python dataset_utils/download_tartanadversity.py
else
  echo "[synthetic] already present — skipping."
fi

# DAWN
if [ ! -f Data/DAWN/dawn_coco.json ]; then
  echo "[dawn] downloading + converting ..."
  python dataset_utils/download_dawn.py
  python dataset_utils/convert_dawn_to_coco.py
else
  echo "[dawn] already present — skipping."
fi

# Foggy Zurich
if [ ! -f Data/FoggyZurich/foggy_zurich_coco.json ]; then
  echo "[foggy_zurich] downloading + preprocessing ..."
  python dataset_utils/download_foggy_zurich.py
  python dataset_utils/preprocess_foggy_zurich.py
else
  echo "[foggy_zurich] already present — skipping."
fi

# --- 4. ACDC preprocessing (only if archives were found) -----------------------
if [ ! -f Data/ACDC/acdc_train_coco.json ]; then
  if [ "$acdc_ready" = true ]; then
    echo "[acdc] preprocessing (extracts fog/rain/snow from the RGB archive) ..."
    python dataset_utils/preprocess_acdc.py
  else
    echo "[acdc] skipped (archives not present)."
  fi
else
  echo "[acdc] already present — skipping."
fi

echo
echo "Setup complete. Configure W&B in .env (see .env.example), then train, e.g.:"
echo "  python train_yolo.py --replacement_percentage 0.2 --augmentation_type gemini"
if [ "$SKIP_ENV" = false ] && [ -d .venv ]; then
  echo "(Remember to 'source .venv/bin/activate' in new shells.)"
fi
