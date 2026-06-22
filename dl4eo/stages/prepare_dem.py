"""
DEM (Copernicus 30m) download and stacking stage.

Downloads DEM tiles once per S2 scene (not per patch), mosaics and
reprojects them to the scene grid, then stacks individual patches with
their scene DEM using a windowed reproject — ~N× faster than per-patch downloading.
"""

import os
import time
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling, transform_bounds
from joblib import Parallel, delayed
import requests


def _tile_name(lat: int, lon: int) -> str:
    lh = "N" if lat >= 0 else "S"
    lo = "E" if lon >= 0 else "W"
    return (
        f"Copernicus_DSM_COG_10_{lh}{abs(lat):02d}_00_"
        f"{lo}{abs(lon):03d}_00_DEM"
    )


def _download_tile(tile_name: str, cache_dir: str):
    out = os.path.join(cache_dir, f"{tile_name}.tif")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    url = (
        f"https://copernicus-dem-30m.s3.amazonaws.com"
        f"/{tile_name}/{tile_name}.tif"
    )
    tmp = out + ".tmp"
    try:
        with requests.get(url, stream=True, timeout=(10, 300)) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    f.write(chunk)
        os.replace(tmp, out)
        print(f"[DEM] Downloaded: {tile_name}")
        return out
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"[WARN] DEM tile unavailable: {tile_name} ({e})")
        return None


def _build_scene_dem(scene_path: str, out_path: str, cache_dir: str, n_jobs: int) -> bool:
    """Download, mosaic, and reproject DEM for one full S2 scene."""
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True

    with rasterio.open(scene_path) as src:
        scene_crs = src.crs
        scene_transform = src.transform
        scene_w, scene_h = src.width, src.height
        wgs84 = transform_bounds(scene_crs, "EPSG:4326", *src.bounds)

    min_lon, min_lat, max_lon, max_lat = wgs84
    tile_names = [
        _tile_name(lat, lon)
        for lat in range(int(np.floor(min_lat)), int(np.ceil(max_lat)))
        for lon in range(int(np.floor(min_lon)), int(np.ceil(max_lon)))
    ]

    os.makedirs(cache_dir, exist_ok=True)
    tile_paths = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_download_tile)(t, cache_dir) for t in tile_names
    )
    tile_paths = [p for p in tile_paths if p is not None]
    if not tile_paths:
        print(f"[ERROR] No DEM tiles available for {os.path.basename(scene_path)}")
        return False

    srcs = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, mosaic_transform = merge(srcs)
        mosaic_crs = srcs[0].crs
    finally:
        for s in srcs:
            s.close()

    dem = np.empty((1, scene_h, scene_w), dtype="float32")
    reproject(
        source=mosaic,
        destination=dem,
        src_transform=mosaic_transform,
        src_crs=mosaic_crs,
        dst_transform=scene_transform,
        dst_crs=scene_crs,
        resampling=Resampling.bilinear,
    )

    meta = {
        "driver": "GTiff", "dtype": "float32", "nodata": None,
        "width": scene_w, "height": scene_h, "count": 1,
        "crs": scene_crs, "transform": scene_transform,
    }
    tmp = out_path + ".tmp"
    with rasterio.open(tmp, "w", **meta) as dst:
        dst.write(dem)
    os.replace(tmp, out_path)
    print(f"[✓] Scene DEM: {os.path.basename(out_path)}")
    return True


def _stack_patch(patch_name: str, patch_dir: str, dem_dir: str, out_dir: str) -> bool:
    """Stack one S2 patch with its scene-level DEM using windowed reproject."""
    out_path = os.path.join(out_dir, patch_name)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[SKIP] Already stacked: {patch_name}")
        return True

    stem = os.path.splitext(patch_name)[0]
    parts = stem.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].isdigit():
        print(f"[ERROR] Cannot parse scene name from: {patch_name}")
        return False
    scene_name = parts[0]

    dem_scene_path = os.path.join(dem_dir, f"{scene_name}_dem.tif")
    patch_path = os.path.join(patch_dir, patch_name)

    if not os.path.exists(dem_scene_path):
        print(f"[ERROR] Scene DEM missing: {dem_scene_path}")
        return False
    if not os.path.exists(patch_path):
        print(f"[ERROR] S2 patch missing: {patch_path}")
        return False

    try:
        with rasterio.open(patch_path) as ps:
            s2 = ps.read().astype("float32")
            patch_crs = ps.crs
            patch_transform = ps.transform
            patch_h, patch_w = ps.height, ps.width
            meta = ps.meta.copy()

        dem_arr = np.empty((patch_h, patch_w), dtype="float32")
        with rasterio.open(dem_scene_path) as ds:
            reproject(
                source=rasterio.band(ds, 1),
                destination=dem_arr,
                src_transform=ds.transform,
                src_crs=ds.crs,
                dst_transform=patch_transform,
                dst_crs=patch_crs,
                resampling=Resampling.bilinear,
            )

        stacked = np.concatenate([s2, dem_arr[np.newaxis]], axis=0)
        meta.update(count=stacked.shape[0], dtype="float32")
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(stacked)
        print(f"[✓] S2+DEM: {patch_name} ({stacked.shape[0]} bands)")
        return True

    except Exception as e:
        print(f"[ERROR] Stack DEM for {patch_name}: {e}")
        return False


def run(cfg):
    if cfg.skip_dem:
        print("[SKIP] DEM stage disabled (skip_dem=True)")
        return

    print("=" * 60)
    print("[START] Preparing DEM")
    start_time = time.time()

    os.makedirs(cfg.dem_dir, exist_ok=True)
    os.makedirs(cfg.stacked_dir, exist_ok=True)
    tile_cache = os.path.join(cfg.dem_dir, "tiles")
    os.makedirs(tile_cache, exist_ok=True)

    # Step 1: one DEM per S2 scene
    scenes = [f for f in os.listdir(cfg.s2_stack) if f.endswith(".tif")]
    print(f"[INFO] Building scene DEMs for {len(scenes)} scene(s)")
    for sf in scenes:
        _build_scene_dem(
            os.path.join(cfg.s2_stack, sf),
            os.path.join(cfg.dem_dir, f"{os.path.splitext(sf)[0]}_dem.tif"),
            tile_cache,
            n_jobs=cfg.n_jobs,
        )

    # Step 2: stack each patch with its scene DEM
    patches = sorted(f for f in os.listdir(cfg.s2_images) if f.endswith(".tif"))
    print(f"[INFO] Stacking {len(patches)} patch(es) with DEM")
    results = Parallel(n_jobs=cfg.n_jobs, prefer="threads")(
        delayed(_stack_patch)(p, cfg.s2_images, cfg.dem_dir, cfg.stacked_dir)
        for p in patches
    )

    n_ok = sum(1 for r in results if r)
    elapsed = time.time() - start_time
    print(f"[DONE] DEM: {n_ok}/{len(patches)} patches stacked in {elapsed:.1f}s")
    print("=" * 60)
