# BacilliNet12: An Enhanced YOLO12 for Bacilli Detection and Counting in Sputum Smear Images

Official implementation of **BacilliNet12**, a lightweight detector for *Mycobacterium tuberculosis* bacilli in Ziehl–Neelsen sputum smear images. It combines a multi-stage preprocessing pipeline (median filter → CLAHE → CIELAB) with YOLO12 Nano, modified with **space-to-depth convolution (SPD-Conv)** in the backbone and an additional high-resolution **P2 detection head**.

> 📄 Paper: *BacilliNet12: An Enhanced YOLO12 with SPD-Conv and P2 Detection Head for Tuberculosis Bacilli Detection and Counting in Sputum Smear Images* (submitted to the International Journal of Intelligent Engineering and Systems, INASS).

---

## What's Included

| Included | Description |
|---|---|
| ✅ Source code | Training, evaluation, and counting scripts |
| ✅ Preprocessing | Median filter → CLAHE → CIELAB pipeline |
| ✅ Modified architecture | BacilliNet12 configuration (YOLO12n + SPD-Conv + P2 head) and the SPD-Conv module |
| ✅ Pre-trained models | Trained BacilliNet12 weights |
| ✉️ Dataset | Available upon reasonable request (see [Dataset](#dataset)) |

## Highlights

- **Multi-stage preprocessing.** A median filter (3 × 3) suppresses impulsive noise, CLAHE (clip limit 2.0, 8 × 8 tiles) equalizes local contrast, and CIELAB conversion emphasizes the red–magenta bacilli on the a\* channel.
- **SPD-Conv backbone.** The first two strided convolutions (layers 0–1) are replaced with SPD-Conv, so fine spatial detail of small bacilli is kept instead of discarded by striding.
- **P2 detection head.** An extra detection head at stride 4 improves the recall of small, faint, and overlapping bacilli.
- **Detection and counting.** Counting is evaluated per image with TP, FP, FN, and mean absolute error (MAE), not only with mAP.
- **Real-time and lightweight.** The model runs at 78.74 FPS (12.70 ms per image, batch size 1) on an RTX 4060 Laptop GPU.

## Results

Test set: 144 images, 1,138 annotated bacilli.

| Model (Full preprocessing) | Precision (%) | Recall (%) | mAP50 (%) | mAP50-95 (%) | FPS | MAE ↓ |
|---|---|---|---|---|---|---|
| YOLO12n | 82.84 | 83.06 | 89.64 | 42.85 | 75.36 | 1.85 |
| YOLO12n + SPD-Conv | 92.50 | 86.54 | 92.05 | 41.80 | 86.51 | 1.22 |
| YOLO12n + P2 head | 90.49 | 94.89 | 91.02 | 42.26 | 77.58 | 0.99 |
| **BacilliNet12 (ours)** | **90.58** | **91.47** | **94.05** | **43.71** | 78.74 | **0.80** |

Additional results:

- **Raw-image baseline.** YOLO12n on raw images (no preprocessing) reaches 84.54% mAP50 with an MAE of 2.32 bacilli per image.
- **Stability.** Five random seeds (0, 42, 128, 512, 1024) give 93.98 ± 0.23% mAP50.
- **Full ablation.** All 24 scenarios (4 architectures × 6 preprocessing variants) are listed in [`results/ablation_24_scenarios.csv`](results/).

## Repository Structure

```
├── configs/
│   ├── bacillinet12.yaml        # model definition (YOLO12n + SPD-Conv + P2 head)
│   └── afb_dataset.yaml         # dataset configuration
├── preprocessing/
│   └── preprocess.py            # median filter → CLAHE → CIELAB
├── modules/
│   └── spdconv.py               # SPD-Conv module
├── scripts/
│   ├── train.py
│   ├── evaluate.py              # P, R, F1, mAP50, mAP50-95, FPS
│   └── evaluate_counting.py     # TP / FP / FN, MAE per image
├── weights/                     # pre-trained models (see below)
├── results/                     # ablation tables, figures
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/samsulchin/BacilliNet12-An-Enhanced-YOLO12-for-Bacilli-Detection-and-Counting-in-Sputum-Smear-Images.git
cd BacilliNet12-An-Enhanced-YOLO12-for-Bacilli-Detection-and-Counting-in-Sputum-Smear-Images
pip install -r requirements.txt
```

Tested with Python 3.x, PyTorch 2.x, and Ultralytics 8.x. See `requirements.txt` for the exact versions.

## Dataset

The dataset contains **1,438** Ziehl–Neelsen sputum smear images with **11,447** bounding-box annotations of a single class (`bacilli`), in YOLO format.

| Split | Images | Bacilli |
|---|---|---|
| Train | 1,150 | 9,177 |
| Validation | 144 | 1,132 |
| Test | 144 | 1,138 |

The dataset is **not publicly distributed** because it consists of clinical images collected from hospital laboratories. It is available from the corresponding author upon reasonable request for research purposes. Please contact us via email with a brief description of your intended use.

If you use your own dataset, arrange it in YOLO format as follows:

```
datasets/afb/
├── train/{images,labels}
├── val/{images,labels}
└── test/{images,labels}
```

## Usage

**1. Preprocess the images**

```bash
python preprocessing/preprocess.py --src datasets/afb --dst datasets/afb_full
```

**2. Train**

```bash
yolo detect train model=configs/bacillinet12.yaml data=configs/afb_dataset.yaml \
     epochs=300 batch=8 imgsz=640 optimizer=Adam patience=40 \
     mosaic=1.0 mixup=0.3 hsv_h=0 hsv_s=0 hsv_v=0 seed=0 deterministic=True
```

**3. Evaluate detection**

```bash
yolo detect val model=weights/bacillinet12_best.pt data=configs/afb_dataset.yaml split=test batch=1
```

**4. Evaluate counting**

```bash
python scripts/evaluate_counting.py --model weights/bacillinet12_best.pt \
       --images datasets/afb_full/test/images --labels datasets/afb_full/test/labels
```

**5. Inference on new images**

```bash
yolo detect predict model=weights/bacillinet12_best.pt source=path/to/preprocessed_images
```

> Note: input images must go through the same preprocessing as the training data (step 1) before inference.

## Pre-trained Models

| Model | Preprocessing | mAP50 (%) | Download |
|---|---|---|---|
| BacilliNet12 | Full | 94.05 | `[ADD LINK]` |

## Citation

If you find this work useful, please cite:

```bibtex
@article{arifin2026bacillinet12,
  title   = {BacilliNet12: An Enhanced YOLO12 with SPD-Conv and P2 Detection Head for Tuberculosis Bacilli Detection and Counting in Sputum Smear Images},
  author  = {Arifin, Samsul and Fatichah, Chastine and Yuniarti, Anny and Yuniarti, Heny and Sigit, Riyanto},
  journal = {International Journal of Intelligent Engineering and Systems},
  year    = {2026},
  note    = {Under review}
}
```

## Acknowledgments

This work builds on [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [YOLO12](https://arxiv.org/abs/2502.12524), and [SPD-Conv](https://github.com/LabSAINT/SPD-Conv).

## License

`[CHOOSE A LICENSE, e.g. AGPL-3.0, to match Ultralytics]`

## Contact

Samsul Arifin — Department of Informatics, Institut Teknologi Sepuluh Nopember (ITS), Surabaya, Indonesia
Email: `[ADD EMAIL]` (for dataset requests)
