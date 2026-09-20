"""Re-evaluate saved checkpoints and write per-epoch metrics to disk.

This is the batch entry point for re-scoring already-trained runs. It performs
*only* evaluation -- no training, no W&B -- and writes one JSON file per
checkpoint. Pushing those files to W&B is a separate, CPU-only step
(``log_eval_results_to_wandb.py``), which keeps the expensive GPU work
independent of the logging and makes re-logging free if anything needs to change.

Evaluation goes through the same code paths the training scripts use:
``Trainer.evaluate`` for ResNet/ConvNeXt and the shared YOLO evaluation utilities
for YOLO. Nothing here reimplements a metric.

Normalization for the Faster R-CNN models is recomputed from the rebuilt training
mixture (seeded, deterministic), never scraped from logs -- the checkpoints do not
store it, and log-scraping is what silently fell back to ImageNet statistics for
at least one run previously.

Examples
--------
    # every checkpoint of one run
    python eval_sweep.py --model_type yolo --replacement_percentage 0.2 \
        --augmentation_type gemini

    # shard a run across 4 shorter jobs (each takes a disjoint slice)
    python eval_sweep.py --model_type convnext --addition_percentage 0.3 \
        --augmentation_type automold --chunk 2 --num_chunks 4
"""
import argparse
import glob
import json
import os
import re
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import ConfigManager


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model_type", required=True, choices=["resnet", "convnext", "yolo"])
    p.add_argument("--replacement_percentage", type=float, default=None)
    p.add_argument("--addition_percentage", type=float, default=None)
    p.add_argument("--augmentation_type", required=True, choices=["gemini", "automold"])
    p.add_argument("--runs_dir", default="runs",
                   help="Directory holding the trained runs (default: runs)")
    p.add_argument("--output_dir", default="eval_results",
                   help="Where per-checkpoint JSON is written (default: eval_results)")
    p.add_argument("--chunk", type=int, default=0,
                   help="0-based index of this shard (default: 0)")
    p.add_argument("--num_chunks", type=int, default=1,
                   help="Total number of shards this run is split across (default: 1)")
    p.add_argument("--force", action="store_true",
                   help="Re-evaluate checkpoints whose JSON already exists")
    p.add_argument("--list_only", action="store_true",
                   help="Print the checkpoints this shard would score, then exit")
    p.add_argument("--norm_cache_dir", default="norm_cache",
                   help="Where per-run normalization statistics are cached "
                        "(Faster R-CNN only; default: norm_cache)")
    p.add_argument("--norm_cache_only", action="store_true",
                   help="Populate the normalization cache for this run and exit. "
                        "Run once per experiment before the GPU sweep so the "
                        "shards do not each redo the full pass over the training "
                        "images.")
    args = p.parse_args()

    if (args.replacement_percentage is None) == (args.addition_percentage is None):
        p.error("Specify exactly one of --replacement_percentage / --addition_percentage")
    if not 0 <= args.chunk < args.num_chunks:
        p.error("--chunk must satisfy 0 <= chunk < num_chunks")
    return args


def eval_schedule(eval_config, total_epochs):
    """Epochs the training config would have evaluated at."""
    from trainer import compute_evaluation_epochs
    return set(compute_evaluation_epochs(eval_config, total_epochs))


def find_rcnn_checkpoints(runs_dir, run_name, max_epoch=None):
    """All saved Faster R-CNN checkpoints, as (epoch, path).

    Every checkpoint on disk is used rather than the config's evaluation
    schedule: these runs were saved every 2 epochs, which does not line up with
    the schedule in configs/{resnet,convnext}.yaml, and filtering by the schedule
    would silently discard most of them.

    ``max_epoch`` bounds the window so every run in a grid is compared over the
    same number of epochs. resnet_r00_gemini was trained to 60 while the rest of
    the ResNet grid stops at 50; without the bound the baseline would get ten
    extra chances at the max-over-epochs each cell reports.
    """
    pattern = os.path.join(runs_dir, run_name, "epoch_*.pth")
    out = []
    for path in glob.glob(pattern):
        m = re.search(r"epoch_(\d+)\.pth$", os.path.basename(path))
        if not m:
            continue
        epoch = int(m.group(1))
        if max_epoch is not None and epoch > max_epoch:
            continue
        out.append((epoch, path))
    return sorted(out)


def find_yolo_checkpoints(runs_dir, run_name, eval_frequency):
    """YOLO checkpoints on the evaluation cadence, as (epoch, path).

    Ultralytics writes ``epochN.pt`` after finishing epoch N+1, so the
    human-readable epoch is N+1 -- the same convention train_yolo.py uses when it
    logs. Every epoch is saved, so this filters to the configured cadence.
    """
    pattern = os.path.join(runs_dir, "yolo", run_name, "weights", "epoch*.pt")
    out = []
    for path in glob.glob(pattern):
        m = re.search(r"epoch(\d+)\.pt$", os.path.basename(path))
        if not m:
            continue
        actual_epoch = int(m.group(1)) + 1
        if actual_epoch % eval_frequency == 0:
            out.append((actual_epoch, path))
    return sorted(out)


def evaluate_rcnn(config_manager, checkpoints, out_dir, force, norm_cache_path=None):
    """Score Faster R-CNN checkpoints with Trainer.evaluate."""
    import torch
    from trainer import BaseTrainer
    from train import build_model
    from data_utils import prepare_synthetic_datasets

    trainer = BaseTrainer(config_manager, wandb_key=None)

    prepared = prepare_synthetic_datasets(config_manager, config_manager.augmentation_type)
    trainer.build_datasets(prepared, norm_cache_path=norm_cache_path)
    # build_datasets only constructs the datasets + normalization; the loaders
    # (including val_loader / additional_val_loaders) come from build_dataloaders.
    # Every training call site pairs the two; the eval path must too, or
    # trainer.evaluate iterates a None loader.
    trainer.build_dataloaders()

    assert trainer.norm_mean is not None and trainer.norm_std is not None, \
        "Normalization stats not computed from the training mixture"
    print(f"[INFO] Normalization recomputed from data: mean={trainer.norm_mean.tolist()} "
          f"std={trainer.norm_std.tolist()}")

    model = build_model(
        config_manager.model_type,
        config_manager.model_config["model"]["num_classes"],
        use_pretrained_weights=config_manager.model_config["model"]["pretrained"],
        image_mean=trainer.norm_mean.tolist(),
        image_std=trainer.norm_std.tolist(),
    )
    trainer.set_model(model)

    for epoch, ckpt_path in checkpoints:
        dest = os.path.join(out_dir, f"epoch_{epoch}.json")
        if os.path.exists(dest) and not force:
            print(f"[SKIP] epoch {epoch}: already evaluated")
            continue

        started = time.time()
        print(f"\n{'='*70}\n[EVAL] {config_manager.run_name} epoch {epoch}\n{'='*70}")
        # weights_only=False: these are our own training checkpoints, and they
        # store numpy scalars (best_map etc.) alongside the state dict. torch>=2.6
        # defaults weights_only=True, which refuses to unpickle those and would
        # crash every Faster R-CNN task on the first checkpoint load.
        ckpt = torch.load(ckpt_path, map_location=trainer.device, weights_only=False)
        trainer.model.load_state_dict(ckpt["model_state_dict"])
        trainer.current_epoch = epoch

        metrics = {}
        for k, v in trainer.evaluate(trainer.val_loader, split_name="val").items():
            if not k.startswith("_"):
                metrics[f"val/{k}"] = float(v)

        for name, loader in (trainer.additional_val_loaders or {}).items():
            print(f"[INFO] Evaluating {name}...")
            for k, v in trainer.evaluate(loader, split_name=name).items():
                if not k.startswith("_"):
                    metrics[f"{name}/{k}"] = float(v)

        write_metrics(dest, config_manager, epoch, ckpt_path, metrics, time.time() - started)


def evaluate_yolo(config_manager, checkpoints, out_dir, force):
    """Score YOLO checkpoints with the shared YOLO evaluation utilities."""
    import torch
    from ultralytics import YOLO
    from train_yolo import YOLOExperimentRunner
    from yolo_utils.yolo_dataset_builder import prepare_temp_dataset
    from yolo_utils.yolo_evaluation_utils import evaluate_additional_datasets

    runner = YOLOExperimentRunner(config_manager)

    # The BDD100K per-weather pass validates against the run's own mixed dataset.
    #
    # It must already exist. Building it here would be wrong in two ways: shards
    # of the same run execute concurrently and would each clear and rebuild the
    # same directory on top of each other, and the mixture was already built
    # during training, so recopying ~30k images per run is pure waste. Stage it
    # with stage_data_from_old_repo.sh.
    temp_base = config_manager.model_config["yolo_temp_dataset_path"]
    temp_dataset_dir = os.path.join(temp_base, config_manager.run_name)
    train_images = os.path.join(temp_dataset_dir, "train", "images")
    if not os.path.isdir(train_images) or not os.listdir(train_images):
        raise RuntimeError(
            f"No prepared training mixture for {config_manager.run_name} at "
            f"{temp_dataset_dir}. Run ./stage_data_from_old_repo.sh to link the "
            f"prebuilt datasets before launching the YOLO sweep -- letting "
            f"concurrent shards build it themselves corrupts the directory."
        )
    data_yaml = prepare_temp_dataset(config_manager, temp_dataset_dir, keep_cached_data=True)

    weather_mapping = runner._create_weather_mapping()

    for epoch, ckpt_path in checkpoints:
        dest = os.path.join(out_dir, f"epoch_{epoch}.json")
        if os.path.exists(dest) and not force:
            print(f"[SKIP] epoch {epoch}: already evaluated")
            continue

        started = time.time()
        print(f"\n{'='*70}\n[EVAL] {config_manager.run_name} epoch {epoch}\n{'='*70}")
        model = YOLO(ckpt_path)

        metrics = {}
        metrics.update(runner._evaluate_per_weather_metrics(model, weather_mapping, data_yaml) or {})
        metrics.update(evaluate_additional_datasets(config_manager, model) or {})
        metrics = {k: float(v) for k, v in metrics.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)}

        write_metrics(dest, config_manager, epoch, ckpt_path, metrics, time.time() - started)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_metrics(dest, config_manager, epoch, ckpt_path, metrics, seconds):
    payload = {
        "run_name": config_manager.run_name,
        "model_type": config_manager.model_type,
        "mixing_method": config_manager.mixing_method,
        "mix_rate": config_manager.mix_rate,
        "augmentation_type": config_manager.augmentation_type,
        "epoch": epoch,
        "checkpoint": os.path.abspath(ckpt_path),
        "eval_seconds": round(seconds, 1),
        "metrics": metrics,
    }
    tmp = dest + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, dest)  # atomic, so a killed job never leaves a partial file
    print(f"[DONE] epoch {epoch}: {len(metrics)} metrics in {seconds:.0f}s -> {dest}")


def set_seed(seed=42):
    """Match the training scripts' seeding so the dataset mixture -- and the
    normalization statistics derived from it -- reproduce exactly."""
    import random
    import numpy as np
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parse_args()
    set_seed(42)

    if args.replacement_percentage is not None:
        mixing_method, rate = "replacement", args.replacement_percentage
    else:
        mixing_method, rate = "addition", args.addition_percentage

    config_manager = ConfigManager(
        "configs",
        model_type=args.model_type,
        mix_rate=rate,
        augmentation_type=args.augmentation_type,
        mixing_method=mixing_method,
    )
    run_name = config_manager.run_name

    norm_cache_path = None
    if args.model_type != "yolo":
        norm_cache_path = os.path.join(args.norm_cache_dir, f"{run_name}.json")

    # Prep mode: build the training mixture once, cache its statistics, stop.
    if args.norm_cache_only:
        if args.model_type == "yolo":
            print("[INFO] YOLO does not use normalization statistics; nothing to do.")
            return 0
        from trainer import BaseTrainer
        from data_utils import prepare_synthetic_datasets
        trainer = BaseTrainer(config_manager, wandb_key=None)
        trainer.build_datasets(
            prepare_synthetic_datasets(config_manager, config_manager.augmentation_type),
            norm_cache_path=norm_cache_path)
        print(f"[DONE] normalization cache ready for {run_name}: {norm_cache_path}")
        return 0

    if args.model_type == "yolo":
        freq = config_manager.model_config["evaluation"]["frequency"]
        checkpoints = find_yolo_checkpoints(args.runs_dir, run_name, freq)
    else:
        checkpoints = find_rcnn_checkpoints(
            args.runs_dir, run_name,
            max_epoch=config_manager.model_config["training"]["epochs"])

    if not checkpoints:
        print(f"[ERROR] No checkpoints found for {run_name} under {args.runs_dir}/")
        return 1

    # Round-robin sharding keeps every shard the same size even when the
    # checkpoint count does not divide evenly.
    shard = checkpoints[args.chunk::args.num_chunks]

    out_dir = os.path.join(args.output_dir, run_name)

    print(f"Run            : {run_name}")
    print(f"Checkpoints    : {len(checkpoints)} total, {len(shard)} in shard "
          f"{args.chunk}/{args.num_chunks}")
    print(f"Epochs in shard: {[e for e, _ in shard]}")
    print(f"Output         : {out_dir}")

    if args.list_only:
        for epoch, path in shard:
            print(f"  epoch {epoch:>4}  {path}")
        return 0

    os.makedirs(out_dir, exist_ok=True)

    if args.model_type == "yolo":
        evaluate_yolo(config_manager, shard, out_dir, args.force)
    else:
        evaluate_rcnn(config_manager, shard, out_dir, args.force, norm_cache_path)

    print(f"\nShard complete: {run_name} [{args.chunk}/{args.num_chunks}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
