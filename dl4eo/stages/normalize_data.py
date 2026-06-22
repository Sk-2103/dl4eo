"""
Normalization stage.

Normalizes patch images to [0, 1] float32.

norm_mode options:
  "per_band"      – independent min-max per band (default, safest for mixed data)
  "per_modality"  – separate percentile-clipped scaling per modality block:
                    optical bands (S2), SAR bands (VV/VH last 2), DEM-derived bands
                    (slope + DEM, second-last 2 before SAR)
  "none"          – skip normalization; copy stacked TIFs as float32
"""

import os
import time
import numpy as np
import rasterio
from joblib import Parallel, delayed


def _per_band_normalize(data: np.ndarray, nodata_val=None) -> np.ndarray:
    """Min-max normalize each band independently, ignoring nodata pixels."""
    out = np.zeros_like(data, dtype="float32")
    for i in range(data.shape[0]):
        band = data[i].astype("float32")
        if nodata_val is not None:
            valid_mask = band != nodata_val
        else:
            valid_mask = ~np.isnan(band)
        valid = band[valid_mask]
        if valid.size == 0:
            continue
        lo, hi = valid.min(), valid.max()
        if hi > lo:
            out[i][valid_mask] = (band[valid_mask] - lo) / (hi - lo)
    return out


def _percentile_normalize(arr: np.ndarray, p_low=2, p_high=98) -> np.ndarray:
    """Clip to [p_low, p_high] percentile then scale to [0, 1]."""
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return arr.astype("float32")
    lo, hi = np.percentile(valid, p_low), np.percentile(valid, p_high)
    if hi <= lo:
        return np.zeros_like(arr, dtype="float32")
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype("float32")


def _per_modality_normalize(data: np.ndarray, n_s2_bands: int) -> np.ndarray:
    """
    Separate percentile-clipped normalization per modality block.
    Expected band order: [s2_bands... | DEM | VV | VH]
    If SAR is absent, the last band is DEM only.
    """
    out = np.zeros_like(data, dtype="float32")
    n = data.shape[0]

    # Detect layout by band count
    has_sar = n == n_s2_bands + 3   # s2 + DEM + VV + VH
    has_dem = n >= n_s2_bands + 1

    if has_sar:
        s2_end = n_s2_bands
        dem_end = n - 2
        sar_start = dem_end
    elif has_dem:
        s2_end = n_s2_bands
        dem_end = n
        sar_start = n
    else:
        s2_end = n
        dem_end = n
        sar_start = n

    # Optical (S2 + index): 2nd–98th percentile
    for i in range(s2_end):
        out[i] = _percentile_normalize(data[i].astype("float32"))

    # DEM: min-max over patch (preserves relative elevation)
    for i in range(s2_end, dem_end):
        band = data[i].astype("float32")
        lo, hi = np.nanmin(band), np.nanmax(band)
        if hi > lo:
            out[i] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)

    # SAR (VV/VH): convert linear power → dB, then 2nd–98th percentile
    for i in range(sar_start, n):
        band = data[i].astype("float32")
        band_db = 10.0 * np.log10(np.clip(band, 1e-10, None))
        out[i] = _percentile_normalize(band_db)

    return out


def run(cfg):
    print("=" * 60)
    print(f"[START] Normalizing data (mode='{cfg.norm_mode}')")
    start_time = time.time()

    if not cfg.normalize or cfg.norm_mode == "none":
        print("[INFO] Normalization skipped per configuration.")
        return

    # Determine source folder in priority order: SAR stack > DEM stack > S2 patches
    def _n_tifs(folder):
        return len([f for f in os.listdir(folder) if f.endswith(".tif")]) if os.path.exists(folder) else 0

    if not cfg.skip_sentinel1 and _n_tifs(cfg.stacked_sar) > 0:
        input_folder = cfg.stacked_sar
    elif not cfg.skip_dem and _n_tifs(cfg.stacked_dir) > 0:
        input_folder = cfg.stacked_dir
    else:
        input_folder = cfg.s2_images

    n_files = _n_tifs(input_folder)
    print(f"[INFO] Normalizing from: {input_folder} ({n_files} file(s))")

    os.makedirs(cfg.normalized, exist_ok=True)

    n_s2 = len(cfg.s2_bands) + (1 if cfg.spectral_index else 0)

    def normalize_file(fname):
        src_path = os.path.join(input_folder, fname)
        dst_path = os.path.join(cfg.normalized, fname)
        if os.path.exists(dst_path):
            print(f"[SKIP] {fname}")
            return
        try:
            with rasterio.open(src_path) as src:
                data = src.read().astype("float32")
                meta = src.meta.copy()
                meta.update(dtype="float32")
                meta.pop("nodata", None)
                nodata_val = src.nodata

            if cfg.norm_mode == "per_band":
                out = _per_band_normalize(data, nodata_val)
            elif cfg.norm_mode == "per_modality":
                out = _per_modality_normalize(data, n_s2)
            else:
                out = data

            with rasterio.open(dst_path, "w", **meta) as dst:
                dst.write(out)
            print(f"[✓] Normalized: {fname}")
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    fnames = [f for f in os.listdir(input_folder) if f.endswith(".tif")]
    Parallel(n_jobs=cfg.n_jobs, prefer="threads")(
        delayed(normalize_file)(f) for f in fnames
    )

    n_out = len([f for f in os.listdir(cfg.normalized) if f.endswith(".tif")])
    elapsed = time.time() - start_time
    print(f"[DONE] Normalized {n_out} file(s) in {elapsed:.1f}s → {cfg.normalized}")
    print("=" * 60)
