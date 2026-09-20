"""Generate the job manifests + sbatch array scripts for the re-evaluation sweep.

Each manifest line is one array task: a (run, shard) pair. Sharding keeps every
job short, because short jobs schedule far sooner than long ones -- a 1-hour
request backfills into gaps that a 9-hour request waits behind.

Run counts and shard sizes are derived from what is actually on disk, not
assumed, so the manifests stay correct if checkpoints are added or removed.

Usage:
    python sbatch_scripts/generate_eval_jobs.py
    python sbatch_scripts/generate_eval_jobs.py --minutes_per_job 45
"""
import argparse
import glob
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "sbatch_scripts", "eval")

# Measured on an L40S from the original sweep: the post-training evaluation of
# yolo_gemini_rep_20 covered 25 checkpoints in ~2h23m across all five eval sets.
# The Faster R-CNN figure is scaled from the DAWN-only offline sweep and is
# deliberately conservative; the pilot run will replace it with a real number.
MINUTES_PER_CHECKPOINT = {"yolo": 6, "resnet": 14, "convnext": 14}

# Must cover each model's configured dataloader workers plus the main process,
# or the workers oversubscribe the cgroup and dataloading throttles everything.
# ROBO nodes are 96 CPUs / 8 GPUs, so 12 still fits 8 tasks per node.
CPUS_PER_TASK = {"yolo": 6, "resnet": 12, "convnext": 10}

# Mix rates that actually have trained runs, per model and mixing method.
REPLACEMENT_RATES = [0, 5, 10, 15, 20, 25, 30, 40, 50]
ADDITION_RATES = [5, 10, 15, 20, 25, 30]
AUGS = ["gemini", "automold"]

SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={model}_eval
#SBATCH --output={out_dir}/logs/{model}_eval_%A_%a.log
#SBATCH --error={out_dir}/logs/{model}_eval_%A_%a.err
#SBATCH --time={walltime}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --account={account}
#SBATCH --array=1-{n_tasks}%{max_concurrent}

# Re-evaluate saved {model} checkpoints. Evaluation only -- no training, and no
# W&B writes; metrics land in eval_results/ and are published separately by
# log_eval_results_to_wandb.py.

# Environment setup must come before `set -euo pipefail`: /etc/bashrc references
# an unbound BASHRCSOURCED and returns non-zero, which trips both -u and -e and
# kills the task in ~3 seconds before anything runs.
source ~/.bashrc
module load anaconda3
conda activate {conda_env}

set -euo pipefail
cd {repo}

MANIFEST="{manifest}"
LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" "$MANIFEST")
if [ -z "$LINE" ]; then
    echo "No manifest entry for array task ${{SLURM_ARRAY_TASK_ID}}" >&2
    exit 1
fi

# manifest columns: mix_flag  rate  augmentation  shard  num_shards
read -r MIX_FLAG RATE AUG SHARD NSHARDS <<< "$LINE"

echo "=== task ${{SLURM_ARRAY_TASK_ID}}: {model} $MIX_FLAG=$RATE $AUG shard $SHARD/$NSHARDS ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python eval_sweep.py \\
    --model_type {model} \\
    --$MIX_FLAG "$RATE" \\
    --augmentation_type "$AUG" \\
    --chunk "$SHARD" \\
    --num_chunks "$NSHARDS"

echo "Finished at: $(date)"
"""


NORM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=norm_cache
#SBATCH --output={out_dir}/logs/norm_cache_%A_%a.log
#SBATCH --error={out_dir}/logs/norm_cache_%A_%a.err
#SBATCH --time={walltime}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --account={account}
#SBATCH --array=1-{n_tasks}%{max_concurrent}

# Precompute per-run normalization statistics for the Faster R-CNN sweeps.
#
# compute_normalization_stats() walks every training image serially, so without
# this the same multi-minute pass would repeat inside each of the ~490 GPU
# shards with the GPU idle. It is pure CPU work, but ROBO rejects jobs that do
# not request a GPU and RM-shared is not available on this account -- so each
# task processes SEVERAL runs back to back, tying up a handful of GPUs briefly
# rather than one per run.

# Environment setup must come before `set -euo pipefail`: /etc/bashrc references
# an unbound BASHRCSOURCED and returns non-zero, which trips both -u and -e and
# kills the task in ~3 seconds before anything runs.
source ~/.bashrc
module load anaconda3
conda activate {conda_env}

set -euo pipefail
cd {repo}

MANIFEST="{manifest}"
TOTAL=$(wc -l < "$MANIFEST")

# Stride through the manifest: task k handles lines k, k+N, k+2N, ...
for i in $(seq "${{SLURM_ARRAY_TASK_ID}}" {n_tasks} "$TOTAL"); do
    LINE=$(sed -n "${{i}}p" "$MANIFEST")
    [ -n "$LINE" ] || continue
    read -r MODEL MIX_FLAG RATE AUG <<< "$LINE"

    echo "=== [$i/$TOTAL] norm cache: $MODEL $MIX_FLAG=$RATE $AUG ==="
    python eval_sweep.py \\
        --model_type "$MODEL" \\
        --$MIX_FLAG "$RATE" \\
        --augmentation_type "$AUG" \\
        --norm_cache_only
done

echo "Finished at: $(date)"
"""


def max_epoch_for(model):
    """Epoch bound eval_sweep.py applies, read from the model config."""
    import yaml
    with open(os.path.join(REPO, "configs", f"{model}.yaml")) as f:
        return yaml.safe_load(f)["training"]["epochs"]


def count_checkpoints(model, run_name):
    """Number of checkpoints this run will actually evaluate.

    Must mirror eval_sweep.py's selection exactly, including the epoch bound,
    or the shard counts here would be computed against a different set than the
    driver ends up slicing.
    """
    if model == "yolo":
        paths = glob.glob(os.path.join(REPO, "runs", "yolo", run_name, "weights", "epoch*.pt"))
        # train_yolo.py evaluates every 10th epoch; epochN.pt is epoch N+1.
        return sum(1 for p in paths
                   if (int(re.search(r"epoch(\d+)\.pt$", os.path.basename(p)).group(1)) + 1) % 10 == 0)
    cap = max_epoch_for(model)
    paths = glob.glob(os.path.join(REPO, "runs", run_name, "epoch_*.pth"))
    return sum(1 for p in paths
               if int(re.search(r"epoch_(\d+)\.pth$", os.path.basename(p)).group(1)) <= cap)


def runs_for(model):
    """Every (mix_flag, rate, aug) that has checkpoints on disk."""
    out = []
    for flag, rates in (("replacement_percentage", REPLACEMENT_RATES),
                        ("addition_percentage", ADDITION_RATES)):
        char = "r" if flag.startswith("replacement") else "a"
        for rate in rates:
            for aug in AUGS:
                run_name = f"{model}_{char}{rate:02d}_{aug}"
                n = count_checkpoints(model, run_name)
                if n:
                    out.append((flag, rate / 100.0, aug, run_name, n))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes_per_job", type=int, default=60,
                    help="Target wall time per array task (default: 60)")
    ap.add_argument("--max_concurrent", type=int, default=20,
                    help="Max simultaneously running array tasks per model")
    ap.add_argument("--conda_env", default="/ocean/projects/cis220039p/jliu43/conda/envs/GenDataEnv")
    ap.add_argument("--account", default="cis220039p",
                    help="Slurm account to charge (default: cis220039p)")
    ap.add_argument("--partition", default="ROBO",
                    help="Slurm partition (default: ROBO -- the only one this "
                         "account's QOS permits)")
    ap.add_argument("--gres", default="gpu:h100:1",
                    help="GPU request (default: gpu:h100:1, the ROBO node type)")
    ap.add_argument("--norm_tasks", type=int, default=8,
                    help="How many parallel tasks share the normalization prep")
    args = ap.parse_args()

    os.makedirs(os.path.join(OUT_DIR, "logs"), exist_ok=True)
    total_jobs = total_gpu_hours = 0

    # --- CPU prep array: one normalization-cache task per Faster R-CNN run ----
    norm_lines = []
    for model in ("resnet", "convnext"):
        for flag, rate, aug, _run_name, _n in runs_for(model):
            norm_lines.append(f"{model} {flag} {rate} {aug}")
    norm_manifest = os.path.join(OUT_DIR, "norm_cache_manifest.txt")
    with open(norm_manifest, "w") as f:
        f.write("\n".join(norm_lines) + "\n")
    norm_script = os.path.join(OUT_DIR, "submit_norm_cache.sh")
    with open(norm_script, "w") as f:
        n_norm = min(args.norm_tasks, len(norm_lines))
        per_task = -(-len(norm_lines) // n_norm)
        norm_walltime = f"{per_task * 25 // 60 + 2:02d}:00:00"
        f.write(NORM_TEMPLATE.format(
            out_dir=OUT_DIR, repo=REPO, manifest=norm_manifest,
            n_tasks=n_norm, max_concurrent=n_norm, walltime=norm_walltime,
            partition=args.partition, gres=args.gres, account=args.account,
            conda_env=args.conda_env))
    os.chmod(norm_script, 0o755)
    print(f"norm-cache {len(norm_lines):3d} runs across {min(args.norm_tasks, len(norm_lines))} "
          f"tasks -- must finish before the resnet/convnext arrays")

    for model in ("yolo", "resnet", "convnext"):
        runs = runs_for(model)
        if not runs:
            print(f"{model}: no checkpoints found under runs/ -- skipped")
            continue

        per_ckpt = MINUTES_PER_CHECKPOINT[model]
        ckpts_per_shard = max(1, args.minutes_per_job // per_ckpt)

        lines, shard_sizes = [], []
        for flag, rate, aug, run_name, n_ckpts in runs:
            n_shards = max(1, -(-n_ckpts // ckpts_per_shard))  # ceil
            for shard in range(n_shards):
                lines.append(f"{flag} {rate} {aug} {shard} {n_shards}")
                shard_sizes.append(len(range(shard, n_ckpts, n_shards)))

        manifest = os.path.join(OUT_DIR, f"{model}_manifest.txt")
        with open(manifest, "w") as f:
            f.write("\n".join(lines) + "\n")

        longest = max(shard_sizes) * per_ckpt
        walltime = f"{(longest * 2) // 60 + 1:02d}:00:00"  # 2x headroom, whole hours

        script = os.path.join(OUT_DIR, f"submit_{model}_eval.sh")
        with open(script, "w") as f:
            f.write(SBATCH_TEMPLATE.format(
                model=model, out_dir=OUT_DIR, repo=REPO, manifest=manifest,
                n_tasks=len(lines), max_concurrent=args.max_concurrent,
                walltime=walltime, partition=args.partition, gres=args.gres,
                account=args.account, cpus=CPUS_PER_TASK[model],
                conda_env=args.conda_env))
        os.chmod(script, 0o755)

        gpu_hours = sum(shard_sizes) * per_ckpt / 60
        total_jobs += len(lines)
        total_gpu_hours += gpu_hours
        print(f"{model:9s} {len(runs):2d} runs, {sum(shard_sizes):4d} checkpoints -> "
              f"{len(lines):3d} tasks of <={max(shard_sizes)} ckpts "
              f"(~{longest} min each, walltime {walltime}, ~{gpu_hours:.0f} GPU-h)")

    print(f"\nTotal: {total_jobs} GPU array tasks, ~{total_gpu_hours:.0f} GPU-hours")
    print(f"Scripts and manifests: {OUT_DIR}")
    print("\nSubmit with (yolo needs no prep; the R-CNN arrays wait on the cache):")
    print(f"  sbatch {os.path.relpath(os.path.join(OUT_DIR, 'submit_yolo_eval.sh'), REPO)}")
    print(f"  NORM=$(sbatch --parsable {os.path.relpath(norm_script, REPO)})")
    for model in ("resnet", "convnext"):
        s = os.path.join(OUT_DIR, f"submit_{model}_eval.sh")
        if os.path.exists(s):
            print(f"  sbatch --dependency=afterok:$NORM {os.path.relpath(s, REPO)}")


if __name__ == "__main__":
    main()
