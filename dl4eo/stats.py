"""
dl4eo.stats — global per-band statistics for a patch dataset.

Usage
-----
    import dl4eo

    stats = dl4eo.stats.compute(
        data_dir="/path/to/output",
        split="train",                  # compute only on training split
        split_file="/path/to/splits.json",
    )
    # → saves /path/to/output/stats.json
    # → returns dict keyed by "band_1" … "band_N"

    # Load later
    stats = dl4eo.stats.load("/path/to/output/stats.json")
"""

import os
import json
import numpy as np
import rasterio
from joblib import Parallel, delayed


def _best_input_dir(data_dir: str) -> str:
    """Return the folder with the most complete stacked patches."""
    def _n(name):
        d = os.path.join(data_dir, name)
        return len([f for f in os.listdir(d) if f.endswith(".tif")]) if os.path.isdir(d) else 0

    for name in ("stacked_with_sar", "stacked", "images"):
        if _n(name) > 0:
            return os.path.join(data_dir, name)
    raise FileNotFoundError(f"No patch TIFs found under {data_dir}")


def _read_bands(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def compute(
    data_dir: str,
    split: str = "train",
    split_file: str = None,
    percentile: tuple = (2, 98),
    n_jobs: int = 4,
) -> dict:
    """
    Compute per-band global statistics across all patches in `split`.

    Parameters
    ----------
    data_dir : str
        Pipeline output directory.
    split : str
        Which split to use ('train' recommended — avoids leakage from val/test).
    split_file : str, optional
        Path to splits.json produced by dl4eo.splits.make_splits().
        If None, uses all available patches.
    percentile : tuple
        (p_low, p_high) for percentile-based clipping stats (default 2, 98).
    n_jobs : int
        Parallel read workers.

    Returns
    -------
    dict  with keys "band_1" … "band_N" and "_meta", saved to {data_dir}/stats.json.
    """
    input_dir = _best_input_dir(data_dir)
    all_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".tif"))

    if split_file and os.path.exists(split_file):
        with open(split_file) as fh:
            splits = json.load(fh)
        stems = set(splits.get(split, []))
        files = [f for f in all_files if os.path.splitext(f)[0] in stems]
        if not files:
            raise ValueError(f"No files matched split='{split}' in {split_file}")
    else:
        files = all_files

    print(f"[INFO] Computing stats from {len(files)} file(s) in {input_dir}")

    # Determine band count from first file
    with rasterio.open(os.path.join(input_dir, files[0])) as src:
        n_bands = src.count

    # Read all patches in parallel
    arrays = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_read_bands)(os.path.join(input_dir, f)) for f in files
    )

    # Aggregate per band (exclude zero / NaN pixels — typical nodata)
    p_lo, p_hi = percentile
    band_stats: dict[str, dict] = {}
    for i in range(n_bands):
        vals = np.concatenate([a[i].ravel() for a in arrays])
        vals = vals[np.isfinite(vals) & (vals != 0)]
        if vals.size == 0:
            band_stats[f"band_{i + 1}"] = {
                "mean": 0.0, "std": 1.0,
                f"p{p_lo}": 0.0, f"p{p_hi}": 1.0,
                "min": 0.0, "max": 1.0,
            }
            continue
        lo, hi = float(np.percentile(vals, p_lo)), float(np.percentile(vals, p_hi))
        band_stats[f"band_{i + 1}"] = {
            "mean": float(vals.mean()),
            "std":  float(vals.std()),
            f"p{p_lo}": lo,
            f"p{p_hi}": hi,
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
        print(f"  Band {i + 1:2d}: mean={band_stats[f'band_{i+1}']['mean']:.4f}  "
              f"std={band_stats[f'band_{i+1}']['std']:.4f}  "
              f"[p{p_lo}={lo:.4f}, p{p_hi}={hi:.4f}]")

    band_stats["_meta"] = {
        "n_files":      len(files),
        "n_bands":      n_bands,
        "split":        split,
        "input_folder": input_dir,
        "percentile":   list(percentile),
    }

    out_path = os.path.join(data_dir, "stats.json")
    with open(out_path, "w") as fh:
        json.dump(band_stats, fh, indent=2)
    print(f"[✓] Stats saved → {out_path}")
    return band_stats


def load(stats_file: str) -> dict:
    """Load statistics previously saved by compute()."""
    with open(stats_file) as fh:
        return json.load(fh)
