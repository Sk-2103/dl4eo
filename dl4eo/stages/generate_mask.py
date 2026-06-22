"""
Mask generation stage.

Rasterizes the feature (label) shapefile to produce a mask matching each
patch's spatial extent and resolution.

Binary mode  (multi_class=False): mask values are 0 (background) or 1 (feature).
Multi-class mode (multi_class=True): mask values are taken from cfg.class_field
  (integer attribute in the shapefile).  Background = 0.
"""

import os
import time
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box
from joblib import Parallel, delayed


def run(cfg):
    print("=" * 60)
    print("[START] Generating segmentation masks")
    start_time = time.time()

    os.makedirs(cfg.masks, exist_ok=True)

    # Load and inspect label shapefile
    gdf_all = gpd.read_file(cfg.lake_shp_path)
    if gdf_all.empty:
        raise RuntimeError(f"Label shapefile is empty: {cfg.lake_shp_path}")

    print(f"[INFO] Label shapefile: {len(gdf_all)} features, CRS={gdf_all.crs}")
    print(f"[INFO] Geometry types: {gdf_all.geom_type.unique().tolist()}")
    print(f"[INFO] Columns: {gdf_all.columns.tolist()}")

    # Determine class values
    if cfg.multi_class:
        field = cfg.class_field
        if field is None:
            # Auto-detect first integer column
            int_cols = [c for c in gdf_all.columns
                        if c != "geometry" and gdf_all[c].dtype in (int, np.int64, np.int32)]
            field = int_cols[0] if int_cols else None
        if field is None or field not in gdf_all.columns:
            raise ValueError(
                "Multi-class mode: specify cfg.class_field with an integer column in the shapefile."
            )
        print(f"[INFO] Multi-class using field '{field}': "
              f"classes {sorted(gdf_all[field].unique().tolist())}")
    else:
        # Binary: ensure an 'id' column exists (value = 1)
        if "id" not in gdf_all.columns:
            gdf_all = gdf_all[[gdf_all.geometry.name]].copy()
            gdf_all["id"] = 1
        field = "id"

    def rasterize_mask(filename):
        if not filename.endswith(".tif"):
            return

        # Use stacked_dir for dimensions; fall back to s2_images
        src_dirs = [cfg.stacked_dir, cfg.s2_images]
        raster_path = None
        for d in src_dirs:
            p = os.path.join(d, filename)
            if os.path.exists(p):
                raster_path = p
                break
        if raster_path is None:
            print(f"[SKIP] Patch raster not found: {filename}")
            return

        output_path = os.path.join(cfg.masks, filename)
        if os.path.exists(output_path):
            print(f"[SKIP] Mask exists: {filename}")
            return

        with rasterio.open(raster_path) as src:
            w, h = src.width, src.height
            transform = src.transform
            crs = src.crs
            bounds = src.bounds

        raster_bbox = box(*bounds)
        gdf_proj = gdf_all.to_crs(crs)
        gdf_clip = gdf_proj[gdf_proj.geometry.intersects(raster_bbox)].copy()
        gdf_clip = gdf_clip[gdf_clip.is_valid & ~gdf_clip.is_empty]

        meta = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 1,
            "dtype": "uint8",
            "crs": crs,
            "transform": transform,
        }

        if gdf_clip.empty:
            # All background
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(np.zeros((1, h, w), dtype="uint8"))
            return

        shapes = [
            (geom, int(val))
            for geom, val in zip(gdf_clip.geometry, gdf_clip[field])
            if geom is not None and geom.is_valid
        ]
        if not shapes:
            with rasterio.open(output_path, "w", **meta) as dst:
                dst.write(np.zeros((1, h, w), dtype="uint8"))
            return

        mask_arr = rasterize(
            shapes=shapes,
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=True,
        )

        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(mask_arr, 1)
        print(f"[✓] Mask: {filename}")

    patch_files = os.listdir(cfg.stacked_dir) if os.path.exists(cfg.stacked_dir) else []
    if not patch_files:
        patch_files = [f for f in os.listdir(cfg.s2_images) if f.endswith(".tif")]

    Parallel(n_jobs=cfg.n_jobs)(delayed(rasterize_mask)(f) for f in patch_files)

    n_masks = len([f for f in os.listdir(cfg.masks) if f.endswith(".tif")])
    elapsed = time.time() - start_time
    print(f"[DONE] Generated {n_masks} mask(s) in {elapsed:.1f}s")
    print("=" * 60)
