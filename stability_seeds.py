"""
Stability test for BacilliNet12: retrain the proposed model with several
random seeds and summarize the results as mean +/- standard deviation.

This script reproduces the seed experiment exactly as it was run for the
paper. Note that, unlike scripts/train.py, it does NOT override Mosaic,
MixUp, or HSV augmentation, so the Ultralytics defaults apply
(mosaic=1.0, mixup=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4).

For each seed, two summaries are produced:
  1. "val"  - the epoch with the highest mAP50-95 in results.csv
              (validation split), as used in the original analysis.
  2. "test" - best.pt re-evaluated on the test split (optional, --eval-test).

Examples:
    python stability_seeds.py --data /path/to/DATASET_FULL_BOOST/data_preprocessed.yaml \
                              --cfg ../configs/bacillinet12.yaml

    python stability_seeds.py --data ... --cfg ... --eval-test
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

TRAIN_ARGS = dict(
    epochs=300,
    patience=40,
    imgsz=640,
    batch=8,
    workers=4,
    deterministic=True,
)

METRIC_COLS = {
    "Precision": "metrics/precision(B)",
    "Recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
}


def find_run_dir(project: str, name: str) -> Optional[Path]:
    """Locate the run folder (Ultralytics versions differ in where they save runs)."""
    for candidate in (Path(project) / name, Path("runs") / "detect" / project / name):
        if (candidate / "results.csv").exists():
            return candidate
    return None


def best_val_epoch(run_dir: Path) -> Dict[str, float]:
    """Return the metrics of the epoch with the highest validation mAP50-95."""
    df = pd.read_csv(run_dir / "results.csv")
    df.columns = df.columns.str.strip()
    row = df.loc[df[METRIC_COLS["mAP50-95"]].idxmax()]
    return {k: float(row[v]) for k, v in METRIC_COLS.items()}


def eval_on_test(run_dir: Path, data: str, device: str) -> Dict[str, float]:
    """Re-evaluate best.pt on the test split."""
    from ultralytics import YOLO

    metrics = YOLO(str(run_dir / "weights" / "best.pt")).val(
        data=data, split="test", batch=1, device=device, verbose=False
    )
    d = metrics.results_dict
    return {k: float(d[v]) for k, v in METRIC_COLS.items()}


def summarize(rows: List[Dict[str, float]], out_csv: Path) -> None:
    """Save per-seed results plus mean and standard deviation (sample SD, ddof=1)."""
    df = pd.DataFrame(rows)
    metric_names = list(METRIC_COLS.keys())
    mean = df[metric_names].mean()
    std = df[metric_names].std()  # pandas default: ddof=1 (sample standard deviation)

    table = df.copy()
    table.loc[len(table)] = {"Seed": "Mean", **mean.to_dict()}
    table.loc[len(table)] = {"Seed": "SD", **std.to_dict()}
    for col in metric_names:
        table[col] = table[col].apply(lambda x: f"{x * 100:.2f}")
    table.to_csv(out_csv, index=False)

    print(f"\n  Saved: {out_csv}")
    for col in metric_names:
        print(f"  {col:<10} {mean[col] * 100:.2f} +/- {std[col] * 100:.2f}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed stability test for BacilliNet12.")
    p.add_argument("--data", required=True, help="Dataset YAML (Full preprocessing).")
    p.add_argument("--cfg", required=True, help="BacilliNet12 architecture YAML.")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 42, 128, 512, 1024])
    p.add_argument("--project", default="runs_seeds", help="Output folder for training runs.")
    p.add_argument("--device", default="0", help="CUDA device, e.g. 0 or cpu.")
    p.add_argument("--eval-test", action="store_true",
                   help="Also re-evaluate each best.pt on the test split.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ultralytics import YOLO

    val_rows: List[Dict[str, float]] = []
    test_rows: List[Dict[str, float]] = []

    print("=" * 60)
    print(f"  Seed stability test: seeds {args.seeds}")
    print("=" * 60)

    for seed in args.seeds:
        name = f"BacilliNet12_seed{seed}"
        print(f"\n [TRAINING] {name}")
        try:
            model = YOLO(args.cfg)  # new model built from scratch for every seed
            model.train(data=args.data, project=args.project, name=name,
                        seed=seed, device=args.device, exist_ok=True, **TRAIN_ARGS)
        except Exception as e:  # keep going with the remaining seeds
            print(f" [ERROR] seed {seed}: {e}")
            continue

        run_dir = find_run_dir(args.project, name)
        if run_dir is None:
            print(f" [WARNING] results.csv for seed {seed} was not found.")
            continue

        val_rows.append({"Seed": seed, **best_val_epoch(run_dir)})
        if args.eval_test:
            test_rows.append({"Seed": seed, **eval_on_test(run_dir, args.data, args.device)})

    print("\n" + "=" * 60)
    print("  Mean and standard deviation")
    print("=" * 60)
    if val_rows:
        print("\n  [Validation split, best epoch by mAP50-95]")
        summarize(val_rows, Path(args.project) / "seeds_summary_val.csv")
    if test_rows:
        print("\n  [Test split, best.pt]")
        summarize(test_rows, Path(args.project) / "seeds_summary_test.csv")
    if not val_rows:
        print("  No results were collected.")


if __name__ == "__main__":
    main()
