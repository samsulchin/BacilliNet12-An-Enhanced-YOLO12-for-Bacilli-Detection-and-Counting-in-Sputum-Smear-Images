"""
Qualitative comparison of detection and counting results for four models
on a single test image (used to produce the qualitative figure in the paper).

Panels:
    (a) YOLOv12 baseline
    (b) YOLOv12 + P2 head
    (c) YOLOv12 + SPD-Conv
    (d) Proposed model (SPD-Conv + P2 head)

Each panel shows the predicted bounding boxes and the total number of
detected bacilli. The input image must already be preprocessed in the same
way as the training data (see preprocessing/preprocess.py).

This file is self-contained (the SPDConv module is defined below).

Example:
    python detect_count_compare.py --image path/to/test/images/Screenshot_74.png \
        --baseline path/to/baseline/best.pt --p2 path/to/p2/best.pt \
        --spd path/to/spd/best.pt --proposed path/to/proposed/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import ultralytics.nn.modules.conv as conv_mod
import ultralytics.nn.tasks as tasks
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv


class SPDConv(nn.Module):
    """Space-to-Depth Convolution. Must be identical to the version used during training."""

    def __init__(self, c1: int, c2: int, dimension: int = 1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(c1 * 4, c2, k=3, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rearrange each 2 x 2 pixel block into the channel dimension, then convolve.
        x = torch.cat([x[..., ::2, ::2],
                       x[..., 1::2, ::2],
                       x[..., ::2, 1::2],
                       x[..., 1::2, 1::2]], 1)
        return self.conv(x)


# Register SPDConv so that Ultralytics can load models containing this module.
conv_mod.SPDConv = SPDConv
tasks.SPDConv = SPDConv


def draw_count(img_bgr, count: int):
    """Write the total count on the image and convert BGR to RGB for Matplotlib."""
    cv2.putText(img_bgr, f"Total Bacilli: {count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3, cv2.LINE_AA)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def compare_four_models(image_path: str, weights: dict, conf: float, out_dir: str) -> Path:
    """Run the four models on one image and save a 2 x 2 comparison figure."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = out_path / f"comparison_{Path(image_path).name}"

    titles = {
        "baseline": "(a) YOLOv12 Baseline",
        "p2":       "(b) YOLOv12 + P2 Head",
        "spd":      "(c) YOLOv12 + SPD-Conv",
        "proposed": "(d) Proposed Model (SPD-Conv + P2)",
    }

    panels = []
    print("Loading models and running inference...")
    for key in ("baseline", "p2", "spd", "proposed"):
        result = YOLO(weights[key]).predict(image_path, conf=conf, verbose=False)[0]
        count = len(result.boxes)
        img_rgb = draw_count(result.plot(line_width=2), count)
        panels.append((img_rgb, f"{titles[key]}\nDetections: {count}"))
        print(f"  {titles[key]:<38} {count} detections")

    # 2 x 2 layout computed from the image size so that the panels are tightly packed
    h_img, w_img = panels[0][0].shape[:2]
    gap = int(0.02 * w_img)                    # gap between panels (2% of image width)
    title_h = int(0.12 * min(h_img, w_img))    # space for the title above each panel
    W = 2 * w_img + gap
    H = 2 * (title_h + h_img) + gap
    dpi_fig = 100
    fig = plt.figure(figsize=(W / dpi_fig, H / dpi_fig), dpi=dpi_fig)
    fs = min(title_h / dpi_fig * 72 * 0.30,                 # fit the title height
             0.95 * w_img / dpi_fig * 72 / (36 * 0.62))     # fit the panel width
    fs = max(fs, 8)

    for k, (img, title) in enumerate(panels):
        r, c = divmod(k, 2)
        x0 = c * (w_img + gap)
        y0 = r * (title_h + h_img + gap)
        ax = fig.add_axes([x0 / W, 1 - (y0 + title_h + h_img) / H, w_img / W, h_img / H])
        ax.imshow(img)
        ax.axis("off")
        fig.text((x0 + w_img / 2) / W, 1 - (y0 + title_h * 0.5) / H, title,
                 ha="center", va="center", fontsize=fs, fontweight="bold")

    fig.savefig(save_path, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {save_path}")
    return save_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare detection and counting of four models on one image.")
    p.add_argument("--image", required=True, help="Path to a preprocessed test image.")
    p.add_argument("--baseline", required=True, help="best.pt of the standard YOLOv12 model.")
    p.add_argument("--p2", required=True, help="best.pt of the YOLOv12 + P2 head model.")
    p.add_argument("--spd", required=True, help="best.pt of the YOLOv12 + SPD-Conv model.")
    p.add_argument("--proposed", required=True, help="best.pt of the proposed BacilliNet12 model.")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25).")
    p.add_argument("--out-dir", default="detection_counting_results", help="Output folder.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    compare_four_models(
        args.image,
        {"baseline": args.baseline, "p2": args.p2, "spd": args.spd, "proposed": args.proposed},
        args.conf,
        args.out_dir,
    )
