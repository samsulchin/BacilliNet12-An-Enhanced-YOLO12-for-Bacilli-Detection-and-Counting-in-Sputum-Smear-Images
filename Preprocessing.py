"""
Multi-stage image preprocessing for BacilliNet12.

Pipeline (applied in this order):
    1. Median filter        - suppresses impulsive noise (dust, stain residue)
                              while preserving bacillus edges.
    2. CLAHE                - equalizes local contrast on the L* (lightness)
                              channel of the CIELAB color space.
    3. CIELAB color boost   - isolates the reddish a* component, keeps only the
                              strongly red bacillus regions via Otsu masking,
                              and enhances those regions in the RGB image.

The script generates the five preprocessed dataset variants used in the
ablation study (the raw dataset is used as-is for the "Raw" scenarios):

    DATASET_MEDIAN            median filter only
    DATASET_CLAHE             CLAHE only
    DATASET_CIELAB_TRANSFORM  CIELAB color boost only
    DATASET_MEDIAN_CLAHE      median filter + CLAHE
    DATASET_FULL_BOOST        median filter + CLAHE + CIELAB color boost (Full)

Expected input structure (YOLO format):

    <src>/
    ├── train/{images,labels}
    ├── val/{images,labels}
    └── test/{images,labels}

Usage:
    python preprocess.py --src DATASET_FIX --out-root .
    python preprocess.py --src DATASET_FIX --scenarios DATASET_FULL_BOOST
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
from tqdm import tqdm

# Supported image extensions and standard YOLO split folders
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val", "test")

# Preprocessing switches for each ablation scenario
SCENARIOS: Dict[str, Dict[str, object]] = {
    "DATASET_MEDIAN":           {"median": True,  "clahe": False, "color": False, "mode": "none"},
    "DATASET_CLAHE":            {"median": False, "clahe": True,  "color": False, "mode": "none"},
    "DATASET_CIELAB_TRANSFORM": {"median": False, "clahe": False, "color": True,  "mode": "boost"},
    "DATASET_MEDIAN_CLAHE":     {"median": True,  "clahe": True,  "color": False, "mode": "none"},
    "DATASET_FULL_BOOST":       {"median": True,  "clahe": True,  "color": True,  "mode": "boost"},
}


def ensure_odd_kernel(value: int) -> int:
    """Return a valid odd kernel size for the median filter (minimum 3)."""
    value = int(value)
    if value < 3:
        return 3
    return value if value % 2 == 1 else value + 1


def preprocess_bacilli_image(
    img_bgr: np.ndarray,
    median_kernel: int = 3,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
    mode: str = "boost",
    use_median: bool = True,
    use_clahe: bool = True,
    use_color_synthesis: bool = True,
) -> np.ndarray:
    """
    Apply the multi-stage preprocessing pipeline to a single BGR image.

    Args:
        img_bgr: Input image in BGR format (as read by OpenCV).
        median_kernel: Median filter kernel size (default 3, i.e., 3 x 3).
        clip_limit: CLAHE contrast limit (default 2.0).
        tile_grid_size: CLAHE grid size; the image is divided into
            tile_grid_size x tile_grid_size tiles (default 8).
        mode: CIELAB color enhancement mode. "boost" is the mode used in all
            reported experiments; "paper" is an alternative channel-synthesis
            mode that was not used in the reported results.
        use_median: Enable stage 1 (median filter).
        use_clahe: Enable stage 2 (CLAHE).
        use_color_synthesis: Enable stage 3 (CIELAB color boost).

    Returns:
        The preprocessed image in BGR format.
    """
    current_img = img_bgr.copy()

    # Stage 1: Median filter - reduces microscopic dust and stain residue
    # without distorting bacillus shape.
    if use_median:
        current_img = cv2.medianBlur(current_img, ensure_odd_kernel(median_kernel))

    # Stage 2: CLAHE on the L* channel - equalizes uneven illumination.
    if use_clahe:
        lab = cv2.cvtColor(current_img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
        )
        l_clahe = clahe.apply(l_ch)
        current_img = cv2.cvtColor(cv2.merge((l_clahe, a_ch, b_ch)), cv2.COLOR_LAB2BGR)

    # Stage 3: CIELAB color boost - robust to different background colors.
    if use_color_synthesis:
        final_lab = cv2.cvtColor(current_img, cv2.COLOR_BGR2LAB)
        _, final_a, _ = cv2.split(final_lab)

        # (a) Isolate the reddish spectrum: in OpenCV, a* = 128 is neutral, so
        #     subtracting 128 keeps only positive (red) values and sets
        #     green/blue tones to 0 (saturating subtraction).
        red_intensity = cv2.subtract(final_a, 128)

        # (b) Otsu dynamic masking: separate strongly red bacilli from faint
        #     residual red stain in the background.
        smoothed_red = cv2.GaussianBlur(red_intensity, (3, 3), 0)
        _, bacilli_mask = cv2.threshold(
            smoothed_red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # (c) Apply the mask: discard red intensity in background regions.
        valid_bacilli_intensity = cv2.bitwise_and(red_intensity, red_intensity, mask=bacilli_mask)

        # (d) Stretch the remaining (bacillus-only) values to the full 0-255 range.
        bacilli_stretched = cv2.normalize(valid_bacilli_intensity, None, 0, 255, cv2.NORM_MINMAX)

        b_ch, g_ch, r_ch = cv2.split(current_img)

        if mode == "boost":
            # bacilli_stretched is exactly 0 in the background, so the boost is
            # applied only to bacillus pixels.
            r_boost = cv2.add(r_ch, bacilli_stretched)
            b_boost = cv2.add(b_ch, bacilli_stretched)

            # Slightly reduce green intensity on bacillus pixels only.
            g_soft = np.where(bacilli_mask > 0, cv2.subtract(g_ch, 50), g_ch)

            current_img = cv2.merge((b_boost, g_soft, r_boost))

        elif mode == "paper":
            # Alternative (not used in the reported results): replace the blue
            # channel with the bacillus intensity on bacillus pixels.
            synthetic_b = np.where(bacilli_mask > 0, bacilli_stretched, b_ch)
            current_img = cv2.merge((synthetic_b, g_ch, r_ch))

    return current_img


def iter_images(folder: Path) -> Iterable[Path]:
    """Return a sorted list of image files in a folder."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def create_data_yaml(output_dir: Path) -> None:
    """Write a YOLO dataset configuration file pointing to the new dataset."""
    yaml_text = (
        f'path: "{output_dir.resolve().as_posix()}"\n'
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n\n"
        "nc: 1\n"
        "names:\n"
        "  0: bacilli\n"
    )
    (output_dir / "data_preprocessed.yaml").write_text(yaml_text, encoding="utf-8")


def process_scenario(input_dir: Path, out_dir: Path, config: Dict[str, object]) -> int:
    """Generate one preprocessed dataset variant. Returns the number of processed images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total_processed = 0

    for split in SPLITS:
        src_img_dir = input_dir / split / "images"
        src_lbl_dir = input_dir / split / "labels"
        if not src_img_dir.exists():
            continue

        dst_img_dir = out_dir / split / "images"
        dst_lbl_dir = out_dir / split / "labels"
        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in tqdm(list(iter_images(src_img_dir)), desc=f"   {split}", unit="img"):
            # Copy the YOLO label file unchanged (preprocessing does not alter geometry).
            lbl_path = src_lbl_dir / img_path.with_suffix(".txt").name
            if lbl_path.exists():
                shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)

            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"   [WARNING] Could not read image: {img_path.name}")
                continue

            processed = preprocess_bacilli_image(
                img_bgr,
                use_median=bool(config["median"]),
                use_clahe=bool(config["clahe"]),
                use_color_synthesis=bool(config["color"]),
                mode=str(config["mode"]),
            )
            cv2.imwrite(str(dst_img_dir / img_path.name), processed)
            total_processed += 1

    create_data_yaml(out_dir)
    return total_processed


def run_all_scenarios(src: str, out_root: str, selected: List[str] | None = None) -> None:
    """Generate the preprocessed dataset variants for the ablation study."""
    input_dir = Path(src).resolve()
    if not input_dir.exists():
        print(f"[ERROR] Input folder '{input_dir}' was not found.")
        return

    names = selected if selected else list(SCENARIOS.keys())
    unknown = [n for n in names if n not in SCENARIOS]
    if unknown:
        print(f"[ERROR] Unknown scenario(s): {', '.join(unknown)}")
        print(f"        Available: {', '.join(SCENARIOS.keys())}")
        return

    print("=" * 60)
    print(f"  Generating {len(names)} preprocessed dataset variant(s)")
    print("=" * 60)

    for name in names:
        print(f"\n-> Scenario: {name}")
        n = process_scenario(input_dir, Path(out_root).resolve() / name, SCENARIOS[name])
        print(f"   [DONE] {n} images processed; data_preprocessed.yaml created.")

    print("\n" + "=" * 60)
    print("  All dataset variants have been generated.")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-stage preprocessing for BacilliNet12.")
    parser.add_argument("--src", default="DATASET_FIX",
                        help="Path to the raw YOLO-format dataset (default: DATASET_FIX).")
    parser.add_argument("--out-root", default=".",
                        help="Folder in which the preprocessed datasets are created (default: current folder).")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        help="Optional subset of scenarios to generate (default: all five).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_all_scenarios(args.src, args.out_root, args.scenarios)
