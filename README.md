# BacilliNet12: An Enhanced YOLO12 for Bacilli Detection and Counting in Sputum Smear Images

This repository contains the code, model configurations, and pretrained weights for BacilliNet12, a lightweight object detector for detecting and counting tuberculosis (TB) bacilli in sputum smear microscopy images.

BacilliNet12 is built on YOLO12 Nano and introduces two architectural modifications:

- SPD-Conv (Space-to-Depth Convolution) replaces the two strided convolutions in the early backbone layers, so that the fine details of thin bacilli are preserved during downsampling instead of being discarded.
- A P2 detection head (stride 4) is added to the three standard heads (P3, P4, P5), giving the model a high-resolution detection level for very small and densely clustered bacilli.

Before training and inference, each image goes through a multi-stage preprocessing pipeline: a median filter to suppress noise, CLAHE on the lightness channel to improve local contrast, and a CIELAB-based color enhancement that emphasizes the reddish bacillus regions.

On the test set, BacilliNet12 with the full preprocessing pipeline reaches 94.05% mAP50 and a counting error (MAE) of 0.80 bacilli per image, while running at about 79 FPS on a single GPU.

This work is part of a master's thesis at the Department of Informatics, Institut Teknologi Sepuluh Nopember (ITS), Surabaya, Indonesia. The accompanying paper is currently under review.


## What Is Included

| Item | Description |
|------|-------------|
| Preprocessing code | Median filter, CLAHE, and CIELAB color enhancement used to build the dataset variants |
| Model configurations | YAML files for the four architectures used in the ablation study |
| Training code | Script for the 24 ablation scenarios and the five-seed stability test |
| Evaluation code | Inference speed benchmark and qualitative detection and counting comparison |
| Pretrained weights | Trained weights of BacilliNet12 and the three comparison models |
| Results | All result tables from the paper in CSV format |

The dataset is not publicly released. See the Dataset section below to request access.


## Repository Structure

```
Configs/
    yolo12n-standard.yaml        Standard YOLO12n (baseline)
    yolo12n-spd.yaml             YOLO12n + SPD-Conv
    yolo12n-p2.yaml              YOLO12n + P2 head
    bacillinet12.yaml            BacilliNet12 (SPD-Conv + P2 head, proposed)
weights/
    bacillinet12_best.pt         BacilliNet12, full preprocessing (S24)
    yolo12n-spd_full_best.pt     YOLO12n + SPD-Conv, full preprocessing (S12)
    yolo12n-p2_full_best.pt      YOLO12n + P2 head, full preprocessing (S18)
    yolo12n_standard.pt          Standard YOLO12n, full preprocessing (S6)
Preprocessing.py                 Builds the preprocessed dataset variants
Training.py                      Trains the 24 ablation scenarios
stability_seeds.py               Retrains BacilliNet12 with five random seeds
benchmark_speed.py               Measures processing time and FPS per scenario
detect_count_compare.py          Compares detection and counting of four models on one image
weights_map_example.yaml         Example mapping of scenarios to weight files
table4_ ... table10_ (.csv)      Result tables from the paper
requirements.txt
CITATION.cff
LICENSE
```


## Installation

The experiments were run on Linux with Python 3.10 or newer. Create an environment and install the dependencies:

```
git clone https://github.com/samsulchin/BacilliNet12-An-Enhanced-YOLO12-for-Bacilli-Detection-and-Counting-in-Sputum-Smear-Images.git
cd BacilliNet12-An-Enhanced-YOLO12-for-Bacilli-Detection-and-Counting-in-Sputum-Smear-Images
pip install -r requirements.txt
```

Install a PyTorch build that matches your CUDA version first if needed (see pytorch.org). Recent GPUs such as the RTX 50 series require a PyTorch build with CUDA 12.8 or newer.


## Dataset

The dataset consists of 1,438 sputum smear microscopy images with 11,447 annotated bacilli in YOLO format, using a single class (bacilli). It is split into training, validation, and test sets; the test set contains 144 images.

The dataset is available on request for research purposes. Please send an email to samsulchin21@gmail.com with your name, affiliation, and a short description of how you intend to use the data.

The scripts expect the YOLO folder structure:

```
DATASET_FIX/
    train/images    train/labels
    val/images      val/labels
    test/images     test/labels
    data.yaml
```

An example data.yaml:

```
path: /path/to/DATASET_FIX
train: train/images
val: val/images
test: test/images
nc: 1
names: ["bacilli"]
```


## Preprocessing

The preprocessing pipeline is applied in the following order:

1. Median filter (3 x 3 kernel) to suppress impulsive noise such as dust and stain residue while keeping bacillus edges.
2. CLAHE (clip limit 2.0, 8 x 8 tiles) on the L* channel of the CIELAB color space to equalize local contrast.
3. CIELAB color enhancement. The reddish a* component is isolated, Otsu thresholding keeps only the strongly red regions, and these regions are enhanced in the RGB image so that the bacilli stand out from the background.

Preprocessing.py generates the five preprocessed dataset variants used in the ablation study. The raw dataset is used as it is for the Raw scenarios.

```
# Generate all five variants next to the raw dataset
python Preprocessing.py --src /path/to/DATASET_FIX --out-root /path/to/datasets

# Generate only the full pipeline (Median + CLAHE + CIELAB)
python Preprocessing.py --src /path/to/DATASET_FIX --out-root /path/to/datasets --scenarios DATASET_FULL_BOOST
```

The output folders are DATASET_MEDIAN, DATASET_CLAHE, DATASET_CIELAB_TRANSFORM, DATASET_MEDIAN_CLAHE, and DATASET_FULL_BOOST. Each contains the processed images, the original labels, and a data_preprocessed.yaml file.

When you use the pretrained weights on new images, apply the same full preprocessing first, because the models were trained on preprocessed images.


## Registering SPD-Conv in Ultralytics

SPD-Conv is not part of the standard Ultralytics package. The inference scripts (benchmark_speed.py and detect_count_compare.py) define and register the module themselves, so the pretrained weights can be loaded without any extra step.

To train the SPD-Conv variants from their YAML files (yolo12n-spd.yaml and bacillinet12.yaml), Ultralytics must be able to build the module. Add the following class to ultralytics/nn/modules/conv.py:

```python
class SPDConv(nn.Module):
    """Space-to-Depth Convolution."""

    def __init__(self, c1, c2, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(c1 * 4, c2, k=3, s=1)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2],
                       x[..., 1::2, ::2],
                       x[..., ::2, 1::2],
                       x[..., 1::2, 1::2]], 1)
        return self.conv(x)
```

Then import SPDConv in ultralytics/nn/tasks.py and add it to the group of modules in parse_model that receive the input and output channels (the same group as Conv), so that its channel arguments are handled like those of a standard convolution.


## Training

All models are built from their YAML files and trained from scratch with the same hyperparameters:

| Parameter | Value |
|-----------|-------|
| Epochs | 300 (early stopping patience 40) |
| Image size | 640 x 640 |
| Batch size | 8 |
| Mosaic | 1.0 |
| MixUp | 0.3 |
| HSV augmentation | disabled (hsv_h, hsv_s, hsv_v = 0) |
| Seed | 0, deterministic mode |

The ablation study combines four architectures with six input variants, giving 24 scenarios:

| Scenarios | Architecture | Config file |
|-----------|--------------|-------------|
| S1 to S6 | Standard YOLO12n | yolo12n-standard.yaml |
| S7 to S12 | YOLO12n + SPD-Conv | yolo12n-spd.yaml |
| S13 to S18 | YOLO12n + P2 head | yolo12n-p2.yaml |
| S19 to S24 | BacilliNet12 (SPD-Conv + P2 head) | bacillinet12.yaml |

Within each group, the six input variants are, in order: Raw, Median, CLAHE, CIELAB, Median + CLAHE, and Full (Median + CLAHE + CIELAB). For example, S24 is BacilliNet12 trained on the fully preprocessed data.

```
# Show the training plan without training
python Training.py --data-root /path/to/datasets --cfg-dir Configs --dry-run

# Train all 24 scenarios
python Training.py --data-root /path/to/datasets --cfg-dir Configs

# Train only the proposed model (S24)
python Training.py --data-root /path/to/datasets --cfg-dir Configs --scenarios S24
```

The folder given with --data-root must contain DATASET_FIX and the five preprocessed variants. Runs are saved under runs_bacillinet12 by default.

To repeat the stability test with five random seeds (0, 42, 128, 512, 1024):

```
python stability_seeds.py --data /path/to/datasets/DATASET_FULL_BOOST/data_preprocessed.yaml --cfg Configs/bacillinet12.yaml
```

This script reproduces the seed experiment as it was run for the paper. Unlike Training.py, it keeps the Ultralytics default augmentation (MixUp off, HSV augmentation on), and by default it summarizes the best epoch on the validation split. Add --eval-test to also evaluate each seed on the test split.

The experiments in the paper were run on an NVIDIA GeForce RTX 5090 (32 GB), an Intel Core Ultra 9 285K, and 64 GB of RAM.


## Inference Speed Benchmark

benchmark_speed.py evaluates each trained model with batch size 1 and records the per-image preprocessing, inference, and postprocessing time reported by Ultralytics. FPS is computed as 1000 divided by the total time in milliseconds. All models are timed on the same fully preprocessed dataset, so that only the architecture affects the result. The image preprocessing pipeline itself is performed beforehand and is not included in the timing.

```
# If your runs were produced by Training.py
python benchmark_speed.py --data /path/to/datasets/DATASET_FULL_BOOST/data_preprocessed.yaml --runs-dir runs_bacillinet12

# If your runs have other names, map each scenario to its weights
python benchmark_speed.py --data /path/to/datasets/DATASET_FULL_BOOST/data_preprocessed.yaml --weights-map weights_map_example.yaml
```

Edit weights_map_example.yaml so that each scenario (S1 to S24) points to the location of its best.pt.


## Detection and Counting Comparison

detect_count_compare.py runs four models on the same image and saves a 2 x 2 figure. Each panel shows the predicted bounding boxes and the total number of detected bacilli. The input image must already be preprocessed with the full pipeline.

```
python detect_count_compare.py \
    --image /path/to/preprocessed_test_image.png \
    --baseline weights/yolo12n_standard.pt \
    --p2 weights/yolo12n-p2_full_best.pt \
    --spd weights/yolo12n-spd_full_best.pt \
    --proposed weights/bacillinet12_best.pt
```

The default confidence threshold is 0.25 and can be changed with --conf. The figure is saved in the detection_counting_results folder.

To use only BacilliNet12 in your own code, load the weights with Ultralytics after registering SPD-Conv (see the SPDConv class at the top of detect_count_compare.py):

```python
from ultralytics import YOLO

model = YOLO("weights/bacillinet12_best.pt")
result = model.predict("preprocessed_image.png", conf=0.25)[0]
print("Detected bacilli:", len(result.boxes))
```


## Results

Main results on the test set (full preprocessing for all models):

| Model | Precision (%) | Recall (%) | mAP50 (%) | mAP50-95 (%) | FPS | MAE (bacilli/image) |
|-------|---------------|------------|-----------|--------------|-----|---------------------|
| Standard YOLO12n (S6) | 82.84 | 83.06 | 89.64 | 42.85 | 75.36 | 1.85 |
| BacilliNet12 (S24) | 90.58 | 91.47 | 94.05 | 43.71 | 78.74 | 0.80 |

Compared with the standard YOLO12n trained on raw images (S1, 84.54% mAP50), the architectural modifications and the preprocessing pipeline together raise mAP50 by 9.51 percentage points.

The complete results are provided as CSV files:

| File | Content |
|------|---------|
| table4_detection_24_scenarios.csv | Detection performance of the 24 scenarios |
| table5_mAP50_contribution.csv | Contribution of the architectural modifications and preprocessing to mAP50 |
| table6_inference_speed.csv | Processing time and inference speed per image |
| table7_seed_stability.csv | Stability across five random seeds |
| table8_counting_five_test_images.csv | Counting results on five test images |
| table9_counting_144_test_images.csv | Breakdown of counting errors on the 144 test images |
| table10_comparison_previous_studies.csv | Comparison with previous studies |


## Citation

If you use this code or the pretrained weights, please cite our work. The citation will be updated when the paper is published.

```
@article{arifin2026bacillinet12,
  title   = {BacilliNet12: An Enhanced YOLO12 with SPD-Conv and P2 Detection Head for Tuberculosis Bacilli Detection and Counting in Sputum Smear Images},
  author  = {Arifin, Samsul and Fatichah, Chastine and Yuniarti, Anny and Yuniarti, Heny and Sigit, Riyanto},
  journal = {International Journal of Intelligent Engineering and Systems},
  year    = {2026},
  note    = {Under review}
}
```

The CITATION.cff file in this repository can also be used through the "Cite this repository" button on GitHub.


## License

This project is released under the GNU Affero General Public License v3.0 (AGPL-3.0), the same license as Ultralytics YOLO. See the LICENSE file for details.


## Acknowledgments

This work builds on Ultralytics YOLO12. The SPD-Conv module follows the design proposed by Sunkara and Luo (2022).


## Contact

For questions about the code, or to request the dataset, please contact Samsul Arifin at samsulchin21@gmail.com.
