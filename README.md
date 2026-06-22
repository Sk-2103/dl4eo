# dl4eo

**dl4eo** is a Python package for building multi-source Earth Observation training datasets and training segmentation models end-to-end. It automates the full pipeline from raw satellite data to evaluated model:

- **Sentinel-2** (L2A, cloud-filtered, spectral indices)
- **Sentinel-1 RTC** (VV + VH, batched by date)
- **Copernicus DEM** (elevation + slope, per-scene mosaic)
- **Segmentation masks** from any vector label file
- **Train-ready PyTorch dataset** with global normalization
- **Model training** with UNet, DeepLabV3+, SegFormer, ViT, and more
- **Evaluation** with per-class IoU / F1 / Precision / Recall / Kappa + GeoTIFF prediction export

---

## Installation

```bash
# Pipeline only (no PyTorch required)
pip install dl4eo

# Pipeline + training + evaluation stack
pip install dl4eo[train]
```

Requires Python ≥ 3.8.

---

## Quick Start

### 1 — Build a dataset

```python
import dl4eo

dl4eo.generate_dataset(
    base_dir="/data/glacial_lakes",
    aoi_shapefile_dir="/data/aoi/",                    # folder containing AOI.shp
    feature_shapefile="/data/lake_boundaries.shp",     # label polygons
    date_range="2021-06-01/2021-08-31",
    cloud_cover=20,
    patch_size=256,
    overlap=0.0,
    spectral_index="NDWI",    # NDWI | NDSI | NDVI | NDRE | EVI | None
    skip_sentinel1=False,
    skip_dem=False,
    sar_days_delta=5,         # search S2_date ± N days for the nearest SAR scene
    normalize=False,          # normalize at load time via PatchDataset instead
    n_jobs=8,
)
```

### 2 — Quality control, splits, statistics

```python
# Filter bad patches (nodata, no foreground, constant bands)
valid = dl4eo.qc.validate("/data/glacial_lakes", min_positive_fraction=0.001)

# Save the valid list
with open("/data/glacial_lakes/valid_patches.txt", "w") as f:
    f.write("\n".join(valid))

# Create train / val / test splits
splits = dl4eo.splits.make_splits(
    "/data/glacial_lakes",
    ratios=(0.7, 0.15, 0.15),
    strategy="temporal",      # "random" | "temporal" | "spatial"
    valid_file="/data/glacial_lakes/valid_patches.txt",
)

# Per-band statistics — training split only, no leakage into val/test
stats = dl4eo.stats.compute("/data/glacial_lakes", split="train")
# returns {"band_1": {"mean": ..., "std": ..., "p2": ..., "p98": ...}, ..., "_meta": {...}}
```

### 3 — PyTorch dataset

```python
from dl4eo.io import PatchDataset
from torch.utils.data import DataLoader

ds = PatchDataset(
    "/data/glacial_lakes",
    split="train",
    split_file="/data/glacial_lakes/splits.json",
    stats_file="/data/glacial_lakes/stats.json",
    norm="zscore",    # "zscore" | "minmax" | "percentile" | None
    bands=None,       # None = all bands; or e.g. [0, 1, 2, 6, 7]
)

sample = ds[0]
# sample["image"]  →  FloatTensor [C, H, W]
# sample["mask"]   →  LongTensor  [H, W]

loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)
```

`PatchDataset` inherits from `torchgeo.datasets.NonGeoDataset` when torchgeo is installed, and falls back to `torch.utils.data.Dataset` otherwise.

### 4 — Train a model (one-liner)

```python
module = dl4eo.train(
    data_dir="/data/glacial_lakes",
    model="unet",             # see Supported Models below
    backbone="resnet34",
    num_classes=2,
    split_strategy="temporal",
    norm="zscore",
    loss="dice_ce",           # "dice_ce" | "dice" | "ce" | "focal"
    batch_size=16,
    max_epochs=50,
    accelerator="gpu",
    devices=1,
    output_dir="/data/checkpoints/unet_run1",
)
# → auto-generates splits.json + stats.json if missing
# → saves best checkpoint monitored on val/iou
# → returns the loaded SegmentationModule
```

### 5 — Evaluate and export predictions

```python
# Option A — pass the module returned from dl4eo.train() directly
report = dl4eo.eval.evaluate(
    module,
    data_dir         = "/data/glacial_lakes",
    splits           = ("val", "test"),
    class_names      = ["background", "lake"],
    output_dir       = "/data/glacial_lakes/eval",
    save_predictions = True,
)

# Option B — reload a checkpoint in a new session
module = dl4eo.eval.load_module(
    "/data/checkpoints/unet_run1/best-epoch=42.ckpt",
    model       = "unet",
    backbone    = "resnet34",
    in_channels = 10,
)
report = dl4eo.eval.evaluate(module, "/data/glacial_lakes",
                              class_names=["background", "lake"])
```

Console output:

```
════════════════════════════════════════════════════════════════════════
  Evaluation — VAL split  (5 patches)
════════════════════════════════════════════════════════════════════════
  ┌──────────────────────┬────────┬────────┬───────────┬────────┬─────────┐
  │        Class         │  IoU   │   F1   │ Precision │ Recall │ Accuracy│
  ├──────────────────────┼────────┼────────┼───────────┼────────┼─────────┤
  │    background (0)    │ 0.9702 │ 0.9849 │   0.9834  │ 0.9863 │    —    │
  │       lake (1)       │ 0.1200 │ 0.2143 │   0.2318  │ 0.1992 │    —    │
  ├──────────────────────┼────────┼────────┼───────────┼────────┼─────────┤
  │     Mean (mIoU)      │ 0.5451 │ 0.5996 │   0.6076  │ 0.5928 │  0.9703 │
  └──────────────────────┴────────┴────────┴───────────┴────────┴─────────┘

  Cohen's Kappa : 0.1992
  Predictions   → /data/glacial_lakes/eval/predictions/val
```

### 6 — Build and train manually (full control)

```python
from dl4eo.train import build_model, SegmentationModule, SegDataModule, SUPPORTED_MODELS
import lightning as L

print(SUPPORTED_MODELS)
# ['unet', 'unet++', 'deeplabv3+', 'fpn', 'pspnet', 'linknet', 'pan', 'manet',
#  'segformer', 'vit-tiny', 'vit-small', 'vit-base']

net    = build_model("segformer", in_channels=10, num_classes=2)
module = SegmentationModule(net, num_classes=2, lr=5e-4, loss="dice_ce")

dm = SegDataModule(
    data_dir   = "/data/glacial_lakes",
    split_file = "/data/glacial_lakes/splits.json",
    stats_file = "/data/glacial_lakes/stats.json",
    batch_size = 8,
)

trainer = L.Trainer(max_epochs=100, accelerator="gpu", devices=1)
trainer.fit(module, dm)
```

---

## Pipeline stages

| Stage | Description |
|-------|-------------|
| 1 | Download Sentinel-2 L2A (STAC / Planetary Computer, cloud-filtered) |
| 2 | Preprocess S2: single-pass resample to 10 m + spectral index + stack |
| 3 | Generate patch AOIs: windowed reads, intersects user AOI polygon |
| 4 | Prepare DEM: one mosaic per scene, windowed reproject per patch |
| 5 | Prepare Sentinel-1 RTC: batched STAC search by date, VV+VH stack |
| 6 | Generate segmentation masks from label shapefile |

Normalization is intentionally excluded from the pipeline. Use `dl4eo.stats.compute()` on the training split and `PatchDataset(norm="zscore")` at load time — this avoids per-patch scale inconsistency and data leakage.

### `PipelineConfig` / `generate_dataset()` parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `base_dir` | — | Output root directory |
| `aoi_shapefile_dir` | — | Folder containing the AOI `.shp` file(s) |
| `feature_shapefile` | — | Full path to the label polygon shapefile |
| `date_range` | — | ISO date range `"YYYY-MM-DD/YYYY-MM-DD"` |
| `cloud_cover` | `20` | Maximum scene cloud cover percentage |
| `patch_size` | `256` | Patch width/height in pixels |
| `overlap` | `0.0` | Fractional overlap between adjacent patches `[0, 1)` |
| `spectral_index` | `"NDWI"` | Spectral index to compute: `"NDWI"`, `"NDSI"`, `"NDVI"`, `"NDRE"`, `"EVI"`, or `None` |
| `sar_days_delta` | `5` | Search window for Sentinel-1: S2 acquisition date ± N days. Increase to 10–15 for high-latitude regions with sparse SAR coverage |
| `skip_sentinel1` | `False` | Skip the SAR stage entirely |
| `skip_dem` | `False` | Skip the DEM stage |
| `normalize` | `False` | Apply min-max normalization in the pipeline (not recommended — use `PatchDataset` instead) |
| `n_jobs` | `8` | Parallel workers for download and processing |

---

## Supported models

All models accept arbitrary `in_channels` and are trained from scratch (no dataset-specific pretrained weights).

| Model | Family | Default backbone | Constraints |
|-------|--------|-----------------|-------------|
| `unet` | SMP | `resnet34` | — |
| `unet++` | SMP | `resnet34` | — |
| `deeplabv3+` | SMP | `resnet34` | `batch_size ≥ 2` per GPU (BatchNorm) |
| `fpn` | SMP | `resnet34` | — |
| `pspnet` | SMP | `resnet34` | `batch_size ≥ 2` per GPU (BatchNorm) |
| `linknet` | SMP | `resnet34` | — |
| `pan` | SMP | `resnet34` | input ≥ 128 px (pyramid pooling) |
| `manet` | SMP | `resnet34` | — |
| `segformer` | SegFormer | auto: `mit_b0` (timm < 1.0) or `swin_tiny_patch4_window7_224` (timm ≥ 1.0) | Any hierarchical timm backbone with `features_only=True` |
| `vit-tiny` | ViT | `vit_tiny_patch16_224` | — |
| `vit-small` | ViT | `vit_small_patch16_224` | — |
| `vit-base` | ViT | `vit_base_patch16_224` | — |

SMP models also support ImageNet-pretrained encoders for 3-channel input: `weights="imagenet"`.

> **SegFormer backbone note:** The original SegFormer architecture (Xie et al., 2021) uses Mix Transformer (MiT) encoders. In timm ≥ 1.0, MiT models were removed. dl4eo auto-detects which backbone is available and selects `mit_b0` (timm < 1.0) or `swin_tiny_patch4_window7_224` (timm ≥ 1.0) as default. You can override this with any timm hierarchical backbone:
> ```python
> # timm < 1.0 — original MiT encoders
> dl4eo.train(model="segformer", backbone="mit_b2", ...)
> # timm ≥ 1.0 — Swin or ConvNeXt
> dl4eo.train(model="segformer", backbone="swin_small_patch4_window7_224", ...)
> dl4eo.train(model="segformer", backbone="convnext_tiny", ...)
> ```

> **BatchNorm note:** `deeplabv3+` and `pspnet` will raise an error if a mini-batch
> contains only 1 sample. Ensure `len(train_set) % batch_size != 1`, or choose a
> `batch_size` that divides your training set evenly.

---

## dl4eo.eval — Evaluation module

### `dl4eo.eval.evaluate()`

Evaluates a trained model on val and/or test splits, prints a metric table, saves GeoTIFF predictions, and writes a full report.

```python
report = dl4eo.eval.evaluate(
    module,                              # SegmentationModule from dl4eo.train()
    data_dir         = "/data/glacial_lakes",
    split_file       = None,             # auto-detected from data_dir/splits.json
    stats_file       = None,             # auto-detected from data_dir/stats.json
    splits           = ("val", "test"),  # which splits to evaluate
    output_dir       = None,             # defaults to data_dir/eval/
    save_predictions = True,             # write per-patch prediction GeoTIFFs
    num_classes      = None,             # auto-inferred from module
    class_names      = None,             # e.g. ["background", "lake"]
    device           = "auto",           # "auto" | "cuda" | "cpu"
)
```

**Returns:** `dict` with keys `"val"` and/or `"test"`, each containing:

```python
{
  "n_patches": 5,
  "per_class": {
    "background": {"iou": 0.9702, "f1": 0.9849, "precision": 0.9834, "recall": 0.9863},
    "lake":       {"iou": 0.1200, "f1": 0.2143, "precision": 0.2318, "recall": 0.1992},
  },
  "mean": {
    "iou": 0.5451, "f1": 0.5996, "precision": 0.6076,
    "recall": 0.5928, "accuracy": 0.9703, "kappa": 0.1992,
  },
  "confusion_matrix": [[...], [...]],   # num_classes × num_classes
}
```

### `dl4eo.eval.load_module()`

Reloads a `SegmentationModule` from a saved `.ckpt` file. Required when evaluating in a new Python session after training.

```python
module = dl4eo.eval.load_module(
    ckpt_path   = "/data/checkpoints/unet/best-epoch=42.ckpt",
    model       = "unet",       # same model name used during training
    backbone    = "resnet34",   # same backbone used during training
    in_channels = 10,           # auto-detected from checkpoint if None
    num_classes = 2,            # auto-detected from checkpoint hparams
)
```

> Because the network architecture is not serialized inside the checkpoint
> (only hyperparameters like `lr`, `loss`, `num_classes` are), you must
> supply the same `model` and `backbone` used during training.

### Metrics

| Metric | Definition |
|--------|-----------|
| **IoU** | Intersection over Union (Jaccard) per class |
| **F1** | 2 · Precision · Recall / (Precision + Recall) per class |
| **Precision** | TP / (TP + FP) — of all pixels predicted as class C, how many are correct |
| **Recall** | TP / (TP + FN) — of all actual class C pixels, how many were found |
| **mIoU** | Mean IoU across all classes |
| **Accuracy** | Overall pixel accuracy — diagonal sum / total pixels |
| **Kappa** | Cohen's Kappa — accuracy corrected for chance agreement |

### Output structure

```
eval/
├── predictions/
│   ├── val/
│   │   ├── S2A_45RXM_20210603_0_L2A_0.tif   ← single-band uint8
│   │   ├── S2A_45RXM_20210603_0_L2A_3.tif   ← same CRS + transform as source patch
│   │   └── ...
│   └── test/
│       └── ...
├── eval_report.json   ← full metrics, confusion matrix, metadata
└── eval_report.txt    ← plain-text table suitable for logs and papers
```

**Prediction GeoTIFFs** are single-band uint8 files (0 = background, 1 = class 1, …) with the exact CRS and affine transform of the corresponding input patch — ready to open in QGIS or overlay with the original imagery.

**`eval_report.txt`** example:

```
========================================================================
  dl4eo Segmentation Evaluation Report
  Generated : 2026-06-19T11:42:25
  Data dir  : /data/glacial_lakes
  Classes   : background, lake
========================================================================

────────────────────────────────────────────────────────────────────────
  Split : VAL   (5 patches)
────────────────────────────────────────────────────────────────────────
  Class                       IoU       F1   Precision   Recall  Accuracy
  ---------------------- -------- -------- ----------- -------- ---------
  background               0.9702   0.9849      0.9834   0.9863         —
  lake                     0.1200   0.2143      0.2318   0.1992         —
  ---------------------- -------- -------- ----------- -------- ---------
  Mean (mIoU)              0.5451   0.5996      0.6076   0.5928    0.9703

  Cohen's Kappa : 0.1992
```

---

## Output structure

```
base_dir/
├── stack/               # Scene-level S2 stacks (bands + spectral index)
├── images/              # Clipped S2 patches
├── DEM/                 # Per-scene DEM mosaics + per-patch stacks
├── GRD/                 # Downloaded SAR granules (VV, VH)
├── Clipped_SAR/         # SAR reprojected to patch grid
├── stacked/             # S2 + DEM patches  (10 bands)
├── stacked_with_sar/    # S2 + DEM + SAR patches  (primary training input)
├── mask/                # Binary (or multi-class) segmentation masks
├── AOI_boxes/           # Per-scene patch grid shapefiles
├── splits.json          # Train / val / test split  (after dl4eo.splits)
├── stats.json           # Per-band statistics        (after dl4eo.stats)
├── valid_patches.txt    # QC-passing patch list      (after dl4eo.qc)
└── eval/                # Evaluation outputs         (after dl4eo.eval)
    ├── predictions/val/
    ├── predictions/test/
    ├── eval_report.json
    └── eval_report.txt
```

---

## Input requirements

| Parameter | Description |
|-----------|-------------|
| `aoi_shapefile_dir` | Folder containing one or more AOI `.shp` files (study area polygon). dl4eo reads all `.shp` files in the folder and unions them. |
| `feature_shapefile` | Full path to the label vector file (e.g. lake outlines) — used for mask generation and patch filtering. |
| `date_range` | `"YYYY-MM-DD/YYYY-MM-DD"` |

The AOI polygon controls which patches are generated. Only patches that intersect both the AOI and at least one label feature are kept.

---

## Dependencies

**Core** (installed automatically):
`numpy`, `rasterio`, `geopandas`, `shapely`, `fiona`, `matplotlib`, `joblib`, `pystac-client`, `planetary-computer`, `requests`, `scipy`

**Training + Evaluation** (`pip install dl4eo[train]`):
`torch>=2.0`, `lightning>=2.0`, `segmentation-models-pytorch>=0.3`, `timm>=0.9`, `torchmetrics>=1.0`

**Optional**:
`torchgeo>=0.5` — enables `NonGeoDataset` base class for `PatchDataset`

---

## Example use cases

- Glacial lake mapping and change detection
- Flood extent extraction from SAR + optical fusion
- Multimodal image segmentation (S2 + S1 + DEM)
- Patch-based training dataset generation for semantic segmentation

---

## Author

Developed by [Saurabh Kaushik](https://scholar.google.com/citations?user=UBGlaXIAAAAJ)
Postdoctoral Researcher · University of Wisconsin–Madison
Earth Observation · Deep Learning · Geo-Foundational Models · Cryosphere

---

## License

MIT License

---

## Citation

If you use `dl4eo` in your research, please cite:

```bibtex
@misc{kaushik2026dl4eo,
  author       = {Saurabh Kaushik},
  title        = {{dl4eo: A Python package for multi-source Earth Observation dataset building and segmentation model training}},
  year         = {2026},
  howpublished = {\url{https://pypi.org/project/dl4eo/}},
}
```
