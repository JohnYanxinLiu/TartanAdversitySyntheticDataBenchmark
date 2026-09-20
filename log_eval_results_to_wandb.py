"""Publish per-checkpoint evaluation JSON to Weights & Biases.

Reads whatever ``eval_sweep.py`` wrote under ``eval_results/`` and creates one
fresh W&B run per experiment, logging the full epoch series in order. CPU-only.

Keeping this separate from the GPU sweep means re-logging costs nothing if a
project, name, or metric mapping needs to change, and a half-finished sweep can
be inspected before anything is published.

It always creates NEW runs. It never resumes an existing run id, so previously
published runs cannot be overwritten or interleaved with.

Usage:
    export WANDB_ENTITY=... WANDB_PROJECT=... WANDB_API_KEY=...
    python log_eval_results_to_wandb.py --dry_run     # inspect first
    python log_eval_results_to_wandb.py
"""
import argparse
import glob
import json
import os
import sys


def load_run(run_dir):
    """Load a run's per-epoch payloads, sorted by epoch."""
    payloads = []
    for path in sorted(glob.glob(os.path.join(run_dir, "epoch_*.json"))):
        with open(path) as f:
            payloads.append(json.load(f))
    payloads.sort(key=lambda p: p["epoch"])
    return payloads


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="eval_results")
    p.add_argument("--entity", default=os.getenv("WANDB_ENTITY"))
    p.add_argument("--project", default=os.getenv("WANDB_PROJECT"))
    p.add_argument("--run", action="append", default=None,
                   help="Only publish these run names (repeatable)")
    p.add_argument("--expected_epochs", type=int, default=None,
                   help="Refuse to publish a run with fewer than this many checkpoints")
    p.add_argument("--dry_run", action="store_true",
                   help="Report what would be published without contacting W&B")
    args = p.parse_args()

    if not args.dry_run and not args.project:
        sys.exit("Set WANDB_PROJECT (and WANDB_ENTITY), or pass --project / --entity.")

    run_dirs = sorted(d for d in glob.glob(os.path.join(args.results_dir, "*"))
                      if os.path.isdir(d))
    if args.run:
        wanted = set(args.run)
        run_dirs = [d for d in run_dirs if os.path.basename(d) in wanted]
    if not run_dirs:
        sys.exit(f"No run directories found under {args.results_dir}/")

    incomplete, published = [], 0
    for run_dir in run_dirs:
        run_name = os.path.basename(run_dir)
        payloads = load_run(run_dir)
        if not payloads:
            print(f"  {run_name:26s} SKIP (no results)")
            continue

        epochs = [p["epoch"] for p in payloads]
        n_metrics = len(payloads[-1]["metrics"])

        if args.expected_epochs is not None and len(payloads) < args.expected_epochs:
            incomplete.append((run_name, len(payloads)))
            print(f"  {run_name:26s} INCOMPLETE {len(payloads)}/{args.expected_epochs} "
                  f"checkpoints -- not published")
            continue

        print(f"  {run_name:26s} {len(payloads):3d} epochs "
              f"({epochs[0]}..{epochs[-1]}), {n_metrics} metrics")

        if args.dry_run:
            continue

        import wandb
        meta = payloads[0]
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            name=run_name,
            reinit=True,
            config={
                "model_type": meta["model_type"],
                "mixing_method": meta["mixing_method"],
                "mix_rate": meta["mix_rate"],
                "augmentation_type": meta["augmentation_type"],
                "source": "eval_sweep.py re-evaluation of saved checkpoints",
            },
        )
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
        for payload in payloads:
            wandb.log({**payload["metrics"], "epoch": payload["epoch"]})
        print(f"      -> {run.url}")
        wandb.finish()
        published += 1

    eligible = len(run_dirs) - len(incomplete)
    if args.dry_run:
        print(f"\nWould publish {eligible} run(s) to "
              f"{args.entity or '<default entity>'}/{args.project or '<unset project>'}.")
    else:
        print(f"\nPublished {published} of {eligible} eligible run(s).")
    if incomplete:
        print("Incomplete runs (re-run their shards before publishing):")
        for name, n in incomplete:
            print(f"  {name}: {n} checkpoints")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
