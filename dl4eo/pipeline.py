"""
DL4EO – main pipeline entry point.

Usage
-----
    import dl4eo

    dl4eo.generate_dataset(
        base_dir="/path/to/output",
        aoi_shapefile_dir="/path/to/aoi/",
        feature_shapefile="/path/to/labels.shp",
        date_range="2021-06-01/2021-08-31",
        cloud_cover=20,
        patch_size=256,
        spectral_index="NDWI",
        overlap=0.0,
        skip_sentinel1=False,
        skip_dem=False,
        n_jobs=8,
    )

    # After the pipeline, compute global stats and train:
    dl4eo.stats.compute("/path/to/output")
    dl4eo.train(
        data_dir="/path/to/output",
        model="unet",
        backbone="resnet34",
        num_classes=2,
    )
"""

import os
import glob
from dl4eo.config import PipelineConfig
from dl4eo.stages import (
    download_sentinel2,
    preprocess_s2,
    generate_aoi,
    prepare_dem,
    prepare_sentinel1,
    generate_mask,
    normalize_data,
)


def generate_dataset(
    base_dir: str,
    aoi_shapefile_dir: str,
    feature_shapefile: str,
    date_range: str,
    cloud_cover: int = 20,
    patch_size: int = 256,
    resolution_m: int = 10,
    overlap: float = 0.0,
    spectral_index: str = "NDWI",
    s2_bands=None,
    skip_sentinel1: bool = False,
    skip_dem: bool = False,
    normalize: bool = False,
    norm_mode: str = "per_band",
    multi_class: bool = False,
    class_field: str = None,
    n_jobs: int = 8,
):
    """
    Build a deep-learning-ready EO segmentation dataset.

    Stages
    ------
    1. Download Sentinel-2 L2A scenes (Element84 Earth Search STAC)
    2. Preprocess S2: resample to 10 m, compute spectral index, stack — all in one pass
    3. Generate patch AOI boxes clipped to the user's AOI polygon
    4. Prepare DEM: mosaic Copernicus 30 m once per scene, stack per patch
    5. Prepare Sentinel-1 RTC (Planetary Computer): batched by date, VV/VH or HH/HV
    6. Generate segmentation masks (binary or multi-class)
    [Optional] Normalize — use dl4eo.stats + dl4eo.io.PatchDataset instead

    Parameters
    ----------
    base_dir : str
        Root directory for all output.
    aoi_shapefile_dir : str
        Directory with AOI polygon shapefile(s). Must NOT contain the feature shapefile.
    feature_shapefile : str
        Path to label polygons (glacial lakes, floods, etc.).
    date_range : str
        ISO-8601 interval, e.g. "2021-06-01/2021-08-31".
    cloud_cover : int
        Max cloud cover % for S2 filtering (default 20).
    patch_size : int
        Patch side length in pixels (default 256).
    resolution_m : int
        Reference resolution in metres (default 10).
    overlap : float
        Fractional patch overlap in [0, 1) (default 0).
    spectral_index : str or None
        One of "NDWI","NDSI","NDVI","NDRE","MNDWI","EVI", or None.
    s2_bands : list of str, optional
        Defaults to ["blue","green","red","nir","swir16","swir22"].
    skip_sentinel1 : bool
        Skip SAR stage (default False).
    skip_dem : bool
        Skip DEM stage (default False).
    normalize : bool
        Write pre-normalised TIFs to disk (default False).
        Prefer dl4eo.stats.compute() + dl4eo.io.PatchDataset for on-the-fly norm.
    norm_mode : str
        "per_band" | "per_modality" | "none" (only used when normalize=True).
    multi_class : bool
        Preserve label class IDs instead of binary masks.
    class_field : str, optional
        Shapefile column with integer class IDs (for multi_class=True).
    n_jobs : int
        Parallel workers (default 8).
    """
    if s2_bands is None:
        from dl4eo.config import DEFAULT_S2_BANDS
        s2_bands = list(DEFAULT_S2_BANDS)

    os.makedirs(base_dir, exist_ok=True)

    cfg = PipelineConfig(
        base_dir=base_dir,
        aoi_shapefile_dir=aoi_shapefile_dir,
        feature_shapefile=feature_shapefile,
        date_range=date_range,
        cloud_cover=cloud_cover,
        s2_bands=s2_bands,
        spectral_index=spectral_index,
        patch_size=patch_size,
        resolution_m=resolution_m,
        overlap=overlap,
        skip_sentinel1=skip_sentinel1,
        skip_dem=skip_dem,
        normalize=normalize,
        norm_mode=norm_mode,
        multi_class=multi_class,
        class_field=class_field,
        n_jobs=n_jobs,
    )

    print(cfg.describe())
    print()

    print("\n[STAGE 1/6] Downloading Sentinel-2")
    download_sentinel2.run(cfg)

    print("\n[STAGE 2/6] Preprocessing Sentinel-2 (single-pass resample+stack)")
    preprocess_s2.run(cfg)

    print("\n[STAGE 3/6] Generating patch AOIs (windowed clipping)")
    generate_aoi.run(cfg)

    print("\n[STAGE 4/6] Preparing DEM")
    prepare_dem.run(cfg)          # internally skips if skip_dem=True

    print("\n[STAGE 5/6] Processing Sentinel-1 SAR")
    prepare_sentinel1.run(cfg)    # internally skips if skip_sentinel1=True

    print("\n[STAGE 6/6] Generating segmentation masks")
    generate_mask.run(cfg)

    if normalize and norm_mode != "none":
        print("\n[POST] Normalizing data")
        normalize_data.run(cfg)

    _print_summary(cfg)


def _print_summary(cfg: PipelineConfig):
    def _count(folder):
        return len(glob.glob(os.path.join(folder, "*.tif")))

    if not cfg.skip_sentinel1 and _count(cfg.stacked_sar) > 0:
        img_dir = cfg.stacked_sar
    elif not cfg.skip_dem and _count(cfg.stacked_dir) > 0:
        img_dir = cfg.stacked_dir
    else:
        img_dir = cfg.s2_images

    n_images = _count(img_dir)
    n_masks  = _count(cfg.masks)

    n_bands = len(cfg.s2_bands) + (1 if cfg.spectral_index else 0)
    band_desc = str(cfg.s2_bands)
    if cfg.spectral_index:
        band_desc += f" + {cfg.spectral_index}"
    if not cfg.skip_dem:
        n_bands += 1
        band_desc += " + DEM"
    if not cfg.skip_sentinel1:
        n_bands += 2
        band_desc += " + VV + VH"

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Images              : {n_images}  →  {img_dir}")
    print(f"  Masks               : {n_masks}   →  {cfg.masks}")
    print(f"  Patch size          : {cfg.patch_size}×{cfg.patch_size} px "
          f"({cfg.box_size_m}m × {cfg.box_size_m}m)")
    print(f"  Bands per patch     : {n_bands}  ({band_desc})")
    if not cfg.normalize:
        print("  Normalization       : off  →  run dl4eo.stats.compute() then "
              "use dl4eo.io.PatchDataset(norm='zscore')")
    print("=" * 60)
