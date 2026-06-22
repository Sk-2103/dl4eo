"""
Sentinel-2 preprocessing stage.

Single-pass: resample all bands to 10 m, compute spectral index, and stack
directly into a multi-band GeoTIFF — no intermediate s2_raw/ directory.
Raw scene folders are deleted after stacking to reclaim disk space.
"""

import os
import time
import shutil
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from joblib import Parallel, delayed
from dl4eo.config import SPECTRAL_INDICES

NATIVE_10M = {"blue", "green", "red", "nir"}


def _process_scene(scene_dir, stack_dir, cfg):
    scene_name = os.path.basename(scene_dir)
    out_path = os.path.join(stack_dir, f"{scene_name}.tif")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        print(f"[SKIP] Stack exists: {scene_name}")
        try:
            shutil.rmtree(scene_dir)
        except Exception:
            pass
        return

    # Map band names to file paths
    band_files = {}
    for fname in os.listdir(scene_dir):
        fl = fname.lower()
        for b in cfg.bands_to_download:
            if fl.endswith(f"_{b.lower()}.tif"):
                band_files[b] = os.path.join(scene_dir, fname)

    missing = [b for b in cfg.bands_to_download if b not in band_files]
    if missing:
        print(f"[WARN] {scene_name}: missing bands {missing}, skipping")
        return

    # Reference grid comes from the green band (native 10 m)
    ref_path = band_files.get("green") or band_files[next(iter(band_files))]

    try:
        with rasterio.open(ref_path) as ref:
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_h, ref_w = ref.height, ref.width
            meta = ref.meta.copy()

        # Resample every band to 10 m in memory
        arrays = {}
        for b, path in band_files.items():
            with rasterio.open(path) as src:
                if b in NATIVE_10M:
                    arrays[b] = src.read(1).astype("float32")
                else:
                    arr = np.empty((ref_h, ref_w), dtype="float32")
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=arr,
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear,
                    )
                    arrays[b] = arr

        # Compute spectral index in memory
        if cfg.spectral_index:
            spec = SPECTRAL_INDICES[cfg.spectral_index]
            if spec is not None:
                b1_name, b2_name = spec
                a1 = arrays[b1_name] / 65535.0
                a2 = arrays[b2_name] / 65535.0
                arrays[cfg.spectral_index] = (a1 - a2) / (a1 + a2 + 1e-8)
            else:  # EVI
                nir  = arrays["nir"]  / 65535.0
                red  = arrays["red"]  / 65535.0
                blue = arrays["blue"] / 65535.0
                arrays[cfg.spectral_index] = np.clip(
                    2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1.0 + 1e-8),
                    -1.0, 1.0,
                )

        band_order = list(cfg.s2_bands)
        if cfg.spectral_index:
            band_order.append(cfg.spectral_index)

        stack = np.stack([arrays[b] for b in band_order], axis=0)
        meta.update(count=len(band_order), dtype="float32")

        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(stack)
            for i, b in enumerate(band_order, 1):
                dst.update_tags(i, name=b)

        print(f"[✓] Stacked {scene_name}: {len(band_order)} bands {band_order}")

    except Exception as e:
        print(f"[ERROR] Processing {scene_name}: {e}")
        return

    # Remove raw scene folder to save space
    try:
        shutil.rmtree(scene_dir)
    except Exception as e:
        print(f"[WARN] Could not delete {scene_dir}: {e}")


def run(cfg):
    print("=" * 60)
    print("[START] Preprocessing Sentinel-2")
    start_time = time.time()

    input_dir = cfg.s2_images   # raw scene subdirs live here after download
    stack_dir = cfg.s2_stack
    os.makedirs(stack_dir, exist_ok=True)

    scene_dirs = [
        os.path.join(input_dir, d)
        for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
        and (d.startswith("S2A") or d.startswith("S2B"))
    ]

    if not scene_dirs:
        print("[INFO] No raw scene directories found (may already be stacked)")
    else:
        print(f"[INFO] Stacking {len(scene_dirs)} scene(s)")
        Parallel(n_jobs=max(1, cfg.n_jobs // 2), prefer="threads")(
            delayed(_process_scene)(d, stack_dir, cfg) for d in scene_dirs
        )

    n_stacked = len([f for f in os.listdir(stack_dir) if f.endswith(".tif")])
    elapsed = time.time() - start_time
    print(f"[DONE] {n_stacked} scene(s) stacked in {elapsed:.1f}s → {stack_dir}")
    print("=" * 60)
