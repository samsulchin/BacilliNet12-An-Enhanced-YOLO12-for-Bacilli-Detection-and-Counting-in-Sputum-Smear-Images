"""
Inference speed benchmark for the 24 ablation scenarios.

For each scenario, the trained best.pt is evaluated with batch size 1, and the
per-image processing time reported by Ultralytics is recorded:
    preprocess  - model-internal preprocessing (resize, normalization)
    inference   - network forward pass
    loss        - loss computation (0 when validating a trained model)
    postprocess - postprocessing, including non-maximum suppression
FPS is computed as 1000 / total time (ms).

All models are timed on the same dataset (Full preprocessing) so that only the
architecture affects latency. The image preprocessing pipeline
(median filter, CLAHE, CIELAB) is performed beforehand and is NOT included.

Weights are located with the run names produced by scripts/train.py
(<runs-dir>/<S#_name>/weights/best.pt). If your runs have other names, pass a
YAML file that maps each scenario to its weights:

    # weights_map.yaml
    S1: path/to/S1/weights/best.pt
    S2: path/to/S2/weights/best.pt
    ...

Examples:
    python benchmark_speed.py --data /path/to/DATASET_FULL_BOOST/data_preprocessed.yaml \
                              --runs-dir runs_bacillinet12
    python benchmark_speed.py --data ... --weights-map weights_map.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
import torch.nn as nn
import ultralytics.nn.modules.conv as conv_mod
import ultralytics.nn.tasks as tasks
import yaml
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv


class SPDConv(nn.Module):
    """Space-to-Depth Convolution. Must be identical to the version used during training."""

    def __init__(self, c1: int, c2: int, dimension: int = 1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(c1 * 4, c2, k=3, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x[..., ::2, ::2],
                       x[..., 1::2, ::2],
                       x[..., ::2, 1::2],
                       x[..., 1::2, 1::2]], 1)
        return self.conv(x)


# Register SPDConv so that Ultralytics can load models containing this module.
conv_mod.SPDConv = SPDConv
tasks.SPDConv = SPDConv

# Scenario names, identical to scripts/train.py
ARCHS = ["YOLO12n", "SPD", "P2", "BacilliNet12"]
INPUTS = ["Raw", "Median", "CLAHE", "CIELAB", "MedianCLAHE", "Full"]
SCENARIOS = {f"S{i * 6 + j + 1}": f"S{i * 6 + j + 1}_{a}_{d}"
             for i, a in enumerate(ARCHS) for j, d in enumerate(INPUTS)}


def resolve_weights(args: argparse.Namespace) -> Dict[str, Path]:
    """Return {scenario id: path to best.pt}."""
    if args.weights_map:
        with open(args.weights_map, encoding="utf-8") as f:
            mapping = yaml.safe_load(f)
        return {sid: Path(p) for sid, p in mapping.items()}
    return {sid: Path(args.runs_dir) / name / "weights" / "best.pt" for sid, name in SCENARIOS.items()}


def benchmark(weights: Path, data: str, imgsz: int, device: str) -> Dict[str, float]:
    """Validate one model with batch size 1 and return its per-image timing."""
    metrics = YOLO(str(weights)).val(data=data, batch=1, imgsz=imgsz, device=device, verbose=False)
    speed = metrics.speed  # ms per image
    pre = speed.get("preprocess", 0.0)
    inf = speed.get("inference", 0.0)
    loss = speed.get("loss", 0.0)
    post = speed.get("postprocess", 0.0)
    total = pre + inf + loss + post
    return {
        "Pre (ms)": round(pre, 2),
        "Infer (ms)": round(inf, 2),
        "Loss (ms)": round(loss, 2),
        "Post (ms)": round(post, 2),
        "Total (ms)": round(total, 2),
        "FPS": round(1000 / total, 2) if total > 0 else 0.0,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark inference speed of the 24 scenarios.")
    p.add_argument("--data", required=True,
                   help="Dataset YAML used for timing (Full preprocessing for all models).")
    p.add_argument("--runs-dir", default="runs_bacillinet12",
                   help="Folder containing the training runs from scripts/train.py.")
    p.add_argument("--weights-map", default=None,
                   help="Optional YAML file mapping S1-S24 to best.pt paths.")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0", help="CUDA device, e.g. 0 or cpu.")
    p.add_argument("--out", default="speed_24_scenarios.csv", help="Output CSV file.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights = resolve_weights(args)
    rows = []

    print("=" * 60)
    print(f"  Inference speed benchmark ({len(weights)} scenarios, batch size 1)")
    print("=" * 60)

    for k, (sid, w) in enumerate(weights.items(), 1):
        print(f"\n[{k}/{len(weights)}] {sid}: {w}")
        if not w.exists():
            print("   [ERROR] Weights not found, skipped.")
            continue
        try:
            r = benchmark(w, args.data, args.imgsz, args.device)
        except Exception as e:  # keep going with the remaining scenarios
            print(f"   [ERROR] {e}")
            continue
        print(f"   inference {r['Infer (ms)']:.2f} ms | total {r['Total (ms)']:.2f} ms | {r['FPS']:.2f} FPS")
        rows.append({"Scenario": sid, **r})

    if rows:
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print("\n" + "=" * 60)
        print(f"  Done. Results saved to: {args.out}")
        print("=" * 60)
    else:
        print("\nNo results were collected.")


if __name__ == "__main__":
    main()
