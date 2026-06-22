"""
dl4eo.qc — patch quality control.

Filters out patches that are unsuitable for training:
  - too much nodata / zero-fill
  - no foreground pixels in the mask
  - all-constant bands (cloud/shadow artifacts)

Usage
-----
    import dl4eo

    valid = dl4eo.qc.validate(
        data_dir="/path/to/output",
        min_valid_fraction=0.8,       # ≥80 % of pixels must be non-zero
        min_positive_fraction=0.001,  # ≥0.1 % of mask pixels must be foreground
        max_nodata_fraction=0.2,
    )
    # → saves /path/to/output/valid_patches.txt
    # → returns list of valid patch stems (without .tif)
"""

import os
import json
import numpy as np
import rasterio
from joblib import Parallel, delayed


def _check_patch(patch_path: str, mask_path: str,
                 min_valid: float, min_pos: float, max_nodata: float):
    stem = os.path.splitext(os.path.basename(patch_path))[0]
    try:
        with rasterio.open(patch_path) as src:
            data = src.read().astype(np.float32)
            nodata = src.nodata

        total_px = data.shape[1] * data.shape[2]

        # Nodata fraction (explicitly flagged or all-zero first band)
        if nodata is not None:
            nodata_count = int(np.sum(data[0] == nodata))
        else:
            nodata_count = int(np.sum(data[0] == 0))
        if nodata_count / total_px > max_nodata:
            return stem, False, "too much nodata"

        # Valid-pixel fraction across ALL bands
        zero_any = np.all(data == 0, axis=0)
        valid_frac = 1.0 - zero_any.mean()
        if valid_frac < min_valid:
            return stem, False, f"valid_fraction={valid_frac:.3f}"

        # Constant-band check (likely cloud/shadow fill)
        for i in range(data.shape[0]):
            band = data[i][~zero_any]
            if band.size > 0 and np.ptp(band) == 0:
                return stem, False, f"band {i+1} is constant"

    except Exception as e:
        return stem, False, f"read error: {e}"

    # Foreground fraction in mask
    if mask_path and os.path.exists(mask_path):
        try:
            with rasterio.open(mask_path) as ms:
                mask = ms.read(1)
            pos_frac = float((mask > 0).mean())
            if pos_frac < min_pos:
                return stem, False, f"positive_fraction={pos_frac:.4f}"
        except Exception:
            pass

    return stem, True, "ok"


def validate(
    data_dir: str,
    min_valid_fraction: float = 0.8,
    min_positive_fraction: float = 0.001,
    max_nodata_fraction: float = 0.2,
    n_jobs: int = 4,
) -> list:
    """
    Validate all patches and return a list of stems that pass all checks.

    Parameters
    ----------
    data_dir : str
        Pipeline output directory.
    min_valid_fraction : float
        Minimum fraction of non-zero pixels across all bands (default 0.8).
    min_positive_fraction : float
        Minimum fraction of foreground pixels in the mask (default 0.001 = 0.1 %).
        Set to 0.0 to skip this check.
    max_nodata_fraction : float
        Maximum allowed nodata pixel fraction (default 0.2).
    n_jobs : int
        Parallel workers.

    Returns
    -------
    list of str  (patch stems without .tif extension)
    """
    # Locate image folder
    for name in ("stacked_with_sar", "stacked", "images"):
        img_dir = os.path.join(data_dir, name)
        files = [f for f in os.listdir(img_dir) if f.endswith(".tif")] if os.path.isdir(img_dir) else []
        if files:
            break
    else:
        raise FileNotFoundError(f"No patch TIFs found under {data_dir}")

    mask_dir = os.path.join(data_dir, "mask")

    print(f"[INFO] Validating {len(files)} patches from {img_dir}")

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_check_patch)(
            os.path.join(img_dir, f),
            os.path.join(mask_dir, f) if os.path.isdir(mask_dir) else None,
            min_valid_fraction, min_positive_fraction, max_nodata_fraction,
        )
        for f in sorted(files)
    )

    valid   = [stem for stem, ok, _ in results if ok]
    invalid = [(stem, reason) for stem, ok, reason in results if not ok]

    print(f"[✓] Valid: {len(valid)} / {len(files)}  "
          f"({len(invalid)} rejected)")
    if invalid:
        for stem, reason in invalid[:10]:
            print(f"    REJECT {stem}: {reason}")
        if len(invalid) > 10:
            print(f"    … and {len(invalid) - 10} more")

    out_path = os.path.join(data_dir, "valid_patches.txt")
    with open(out_path, "w") as fh:
        fh.write("\n".join(valid))
    print(f"[✓] Valid patch list → {out_path}")

    return valid
