"""
dl4eo.io — PyTorch dataset for dl4eo patch outputs.

Returned dict format (torchgeo-compatible)
------------------------------------------
    {"image": FloatTensor[C, H, W], "mask": LongTensor[H, W]}

Usage
-----
    from dl4eo.io import PatchDataset

    ds = PatchDataset(
        data_dir="/path/to/output",
        split="train",
        split_file="/path/to/output/splits.json",
        stats_file="/path/to/output/stats.json",
        norm="zscore",        # "zscore" | "minmax" | "percentile" | None
        bands=None,           # None = all bands; or e.g. [0,1,2,6,7] (0-indexed)
    )

    sample = ds[0]
    image  = sample["image"]   # FloatTensor [C, H, W]
    mask   = sample["mask"]    # LongTensor  [H, W]

    # PyTorch DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=16, shuffle=True, num_workers=4)
"""

import os
import json
import numpy as np
import rasterio

try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    # Try torchgeo for compatibility; fall back to plain Dataset
    try:
        from torchgeo.datasets import NonGeoDataset as _Base
    except ImportError:
        _Base = _TorchDataset
except ImportError:
    raise ImportError(
        "PyTorch is required for dl4eo.io.  "
        "Install it with:  pip install torch"
    )


def _best_img_dir(data_dir: str) -> str:
    for name in ("stacked_with_sar", "stacked", "images"):
        d = os.path.join(data_dir, name)
        if os.path.isdir(d) and any(f.endswith(".tif") for f in os.listdir(d)):
            return d
    raise FileNotFoundError(f"No patch TIFs found under {data_dir}")


class PatchDataset(_Base):
    """
    Dataset of dl4eo GeoTIFF patches and binary/multi-class masks.

    Parameters
    ----------
    data_dir : str
        Pipeline output directory.
    split : str
        "train" | "val" | "test" (default "train").
    split_file : str, optional
        Path to splits.json.  If None, all patches are used.
    stats_file : str, optional
        Path to stats.json from dl4eo.stats.compute().  Required when norm != None.
    norm : str or None
        "zscore"     – (x − mean) / std  per band
        "minmax"     – (x − min)  / (max − min) per band
        "percentile" – clip to [p2, p98] then scale to [0, 1]
        None         – return raw float32 values
    bands : list of int, optional
        0-indexed band indices to return.  None = all bands.
    transforms : callable, optional
        Extra transform applied to the returned dict after normalization.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        split_file: str = None,
        stats_file: str = None,
        norm: str = "zscore",
        bands: list = None,
        transforms=None,
    ):
        self.data_dir   = data_dir
        self.split      = split
        self.norm       = norm
        self.bands      = bands
        self.transforms = transforms

        self.img_dir  = _best_img_dir(data_dir)
        self.mask_dir = os.path.join(data_dir, "mask")

        # Resolve file list
        all_files = sorted(f for f in os.listdir(self.img_dir) if f.endswith(".tif"))
        if split_file and os.path.exists(split_file):
            with open(split_file) as fh:
                splits = json.load(fh)
            stems = set(splits.get(split, []))
            self.files = [f for f in all_files if os.path.splitext(f)[0] in stems]
        else:
            self.files = all_files

        if not self.files:
            raise ValueError(
                f"No files found for split='{split}' in {self.img_dir}. "
                "Run dl4eo.splits.make_splits() first."
            )

        # Load stats for normalization
        self._stats = None
        if norm is not None:
            if stats_file is None:
                stats_file = os.path.join(data_dir, "stats.json")
            if os.path.exists(stats_file):
                with open(stats_file) as fh:
                    self._stats = json.load(fh)
            else:
                raise FileNotFoundError(
                    f"stats.json not found at {stats_file}. "
                    "Run dl4eo.stats.compute() first."
                )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        fname = self.files[idx]

        with rasterio.open(os.path.join(self.img_dir, fname)) as src:
            image = src.read().astype(np.float32)  # [C, H, W]

        # Band selection
        if self.bands is not None:
            image = image[self.bands]

        # Normalization
        if self.norm is not None and self._stats is not None:
            image = self._normalize(image)

        # Mask
        mask_path = os.path.join(self.mask_dir, fname)
        if os.path.exists(mask_path):
            with rasterio.open(mask_path) as ms:
                mask = ms.read(1).astype(np.int64)
        else:
            mask = np.zeros(image.shape[1:], dtype=np.int64)

        sample = {
            "image": torch.from_numpy(image),
            "mask":  torch.from_numpy(mask),
        }

        if self.transforms:
            sample = self.transforms(sample)

        return sample

    # ------------------------------------------------------------------
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        out = np.empty_like(image)
        for i in range(image.shape[0]):
            band_idx = (self.bands[i] if self.bands else i) + 1
            key = f"band_{band_idx}"
            s   = self._stats.get(key, {})

            if self.norm == "zscore":
                mean = s.get("mean", 0.0)
                std  = s.get("std",  1.0)
                out[i] = (image[i] - mean) / (std + 1e-8)

            elif self.norm == "minmax":
                lo = s.get("min", 0.0)
                hi = s.get("max", 1.0)
                out[i] = np.clip((image[i] - lo) / (hi - lo + 1e-8), 0.0, 1.0)

            elif self.norm == "percentile":
                # Find percentile keys dynamically
                keys = [k for k in s if k.startswith("p") and k[1:].isdigit()]
                lo_key = sorted(keys, key=lambda k: int(k[1:]))[0] if keys else "p2"
                hi_key = sorted(keys, key=lambda k: int(k[1:]))[-1] if keys else "p98"
                lo = s.get(lo_key, 0.0)
                hi = s.get(hi_key, 1.0)
                out[i] = np.clip((image[i] - lo) / (hi - lo + 1e-8), 0.0, 1.0)

            else:
                out[i] = image[i]
        return out

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"PatchDataset(split='{self.split}', n={len(self)}, "
            f"norm='{self.norm}', img_dir='{self.img_dir}')"
        )
