"""
Training script for the BacilliNet12 ablation study (24 scenarios).

Scenario design (see Table 3 in the paper):

    S1  - S6   Standard YOLO12n                 x 6 input variants
    S7  - S12  YOLO12n + SPD-Conv               x 6 input variants
    S13 - S18  YOLO12n + P2 head                x 6 input variants
    S19 - S24  BacilliNet12 (SPD-Conv + P2)     x 6 input variants

Input variants (in this order): Raw, Median, CLAHE, CIELAB, Median + CLAHE,
Full (Median + CLAHE + CIELAB). The preprocessed variants are generated with
preprocessing/preprocess.py.

All models are built from their YAML definitions and trained from scratch
with identical hyperparameters.

Examples:
    # Train all 24 scenarios
    python train.py --data-root /path/to/datasets --cfg-dir ../configs

    # Train only the proposed model (S24)
    python train.py --data-root /path/to/datasets --cfg-dir ../configs --scenarios S24

    # Stability test: retrain S24 with five random seeds
    python train.py --data-root /path/to/datasets --cfg-dir ../configs \
                    --scenarios S24 --seeds 0 42 128 512 1024

    # Show the training plan without training
    python train.py --data-root /path/to/datasets --cfg-dir ../configs --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# -----------------------------------------------------------------------------
# Training hyperparameters (identical for all scenarios)
# -----------------------------------------------------------------------------
TRAIN_ARGS = dict(
    epochs=300,
    patience=40,        # early stopping
    imgsz=640,
    batch=8,
    mosaic=1.0,
    mixup=0.3,
    hsv_h=0.0,          # HSV augmentation disabled so that the bacilli colors
    hsv_s=0.0,          # enhanced by preprocessing are not randomly altered
    hsv_v=0.0,
    workers=4,
    deterministic=True,
)

# -----------------------------------------------------------------------------
# Architectures and input variants
# Adjust the file/folder names below to match your own setup.
# -----------------------------------------------------------------------------
ARCHITECTURES: List[Tuple[str, str]] = [
    # (short name, architecture YAML file inside --cfg-dir)
    ("YOLO12n",      "yolo12n-standard.yaml"),
    ("SPD",          "yolo12n-spd.yaml"),
    ("P2",           "yolo12n-p2.yaml"),
    ("BacilliNet12", "bacillinet12.yaml"),
]

INPUT_VARIANTS: List[Tuple[str, str]] = [
    # (short name, dataset YAML path relative to --data-root)
    ("Raw",         "DATASET_FIX/data.yaml"),
    ("Median",      "DATASET_MEDIAN/data_preprocessed.yaml"),
    ("CLAHE",       "DATASET_CLAHE/data_preprocessed.yaml"),
    ("CIELAB",      "DATASET_CIELAB_TRANSFORM/data_preprocessed.yaml"),
    ("MedianCLAHE", "DATASET_MEDIAN_CLAHE/data_preprocessed.yaml"),
    ("Full",        "DATASET_FULL_BOOST/data_preprocessed.yaml"),
]


def build_scenarios(data_root: Path, cfg_dir: Path) -> Dict[str, Dict[str, str]]:
    """Map S1-S24 to (architecture YAML, dataset YAML, run name)."""
    scenarios: Dict[str, Dict[str, str]] = {}
    idx = 1
    for arch_name, arch_yaml in ARCHITECTURES:
        for data_name, data_yaml in INPUT_VARIANTS:
            sid = f"S{idx}"
            scenarios[sid] = {
                "model": str(cfg_dir / arch_yaml),
                "data": str(data_root / data_yaml),
                "name": f"{sid}_{arch_name}_{data_name}",
            }
            idx += 1
    return scenarios


def precheck(selected: Dict[str, Dict[str, str]]) -> None:
    """Make sure all dataset and architecture files exist before training starts."""
    missing = []
    for sid, sc in selected.items():
        if not Path(sc["data"]).exists():
            missing.append(f"[{sid}] dataset: {sc['data']}")
        if not Path(sc["model"]).exists():
            missing.append(f"[{sid}] architecture: {sc['model']}")
    if missing:
        print("\n[ERROR] The following files were not found:")
        for m in missing:
            print("   -", m)
        raise FileNotFoundError("Fix the paths above before running the training.")
    print(">> Pre-check passed: all dataset and architecture files were found.\n")


def train_scenario(sc: Dict[str, str], project: str, seed: int, device: str) -> None:
    """Train one scenario from scratch, skipping it if it has already been completed."""
    from ultralytics import YOLO  # imported here so that --dry-run works without Ultralytics

    run_name = sc["name"] if seed == 0 else f"{sc['name']}_seed{seed}"
    if (Path(project) / run_name / "weights" / "last.pt").exists():
        print(f" [SKIP] '{run_name}' has already been trained.")
        return

    print(f"\n [TRAINING] {run_name}")
    print(f"            model: {sc['model']}")
    print(f"            data : {sc['data']}")

    model = YOLO(sc["model"])  # build the architecture from its YAML definition (from scratch)
    model.train(
        data=sc["data"],
        project=project,
        name=run_name,
        exist_ok=True,
        seed=seed,
        device=device,
        **TRAIN_ARGS,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the BacilliNet12 ablation scenarios.")
    p.add_argument("--data-root", required=True,
                   help="Folder containing the raw and preprocessed dataset variants.")
    p.add_argument("--cfg-dir", required=True,
                   help="Folder containing the architecture YAML files.")
    p.add_argument("--scenarios", nargs="+", default=["all"],
                   help="Scenarios to train, e.g. S1 S6 S24 (default: all).")
    p.add_argument("--seeds", nargs="+", type=int, default=[0],
                   help="Random seed(s); the reported main results use seed 0.")
    p.add_argument("--project", default="runs_bacillinet12",
                   help="Output folder for training runs.")
    p.add_argument("--device", default="0", help="CUDA device, e.g. 0 or cpu.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the training plan without training.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    all_scenarios = build_scenarios(Path(args.data_root), Path(args.cfg_dir))

    if args.scenarios == ["all"]:
        selected = all_scenarios
    else:
        unknown = [s for s in args.scenarios if s not in all_scenarios]
        if unknown:
            raise ValueError(f"Unknown scenario(s): {', '.join(unknown)} (valid: S1-S24)")
        selected = {s: all_scenarios[s] for s in args.scenarios}

    print("=" * 60)
    print(f"  BacilliNet12 training: {len(selected)} scenario(s) x {len(args.seeds)} seed(s)")
    print("=" * 60)
    for sid, sc in selected.items():
        print(f"  {sid:<4} {sc['name']}")

    if args.dry_run:
        return

    precheck(selected)
    for seed in args.seeds:
        for sc in selected.values():
            train_scenario(sc, args.project, seed, args.device)

    print("\n" + "=" * 60)
    print("  All selected scenarios have been trained.")
    print("=" * 60)


if __name__ == "__main__":
    main()
