"""
dl4eo.splits — train / val / test split strategies.

Three strategies
----------------
random   : random shuffle then split by ratio  (fast baseline)
temporal : split by acquisition date — avoids temporal leakage across splits
spatial  : split by S2 tile code — avoids spatial leakage (strictest)

Usage
-----
    import dl4eo

    splits = dl4eo.splits.make_splits(
        data_dir="/path/to/output",
        ratios=(0.7, 0.15, 0.15),
        strategy="temporal",
        valid_file="/path/to/output/valid_patches.txt",  # optional
        seed=42,
    )
    # → saves /path/to/output/splits.json
    # → returns {"train": [...], "val": [...], "test": [...]}

    # Load later
    splits = dl4eo.splits.load("/path/to/output/splits.json")
"""

import os
import re
import json
import random
from collections import defaultdict


def _stems_from_dir(data_dir: str, valid_file: str = None) -> list:
    """Return patch stems (without .tif) from the best available folder."""
    for name in ("stacked_with_sar", "stacked", "images"):
        d = os.path.join(data_dir, name)
        if os.path.isdir(d):
            files = [f for f in os.listdir(d) if f.endswith(".tif")]
            if files:
                stems = [os.path.splitext(f)[0] for f in sorted(files)]
                break
    else:
        raise FileNotFoundError(f"No patch TIFs found under {data_dir}")

    if valid_file and os.path.exists(valid_file):
        with open(valid_file) as fh:
            allowed = set(line.strip() for line in fh if line.strip())
        stems = [s for s in stems if s in allowed]
        print(f"[INFO] Filtered to {len(stems)} valid patches")

    return stems


def _assign_ratios(items: list, ratios: tuple, rng: random.Random) -> dict:
    """Shuffle and assign items to train/val/test by ratio."""
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = round(ratios[0] * n)
    n_val   = round(ratios[1] * n)
    return {
        "train": shuffled[:n_train],
        "val":   shuffled[n_train: n_train + n_val],
        "test":  shuffled[n_train + n_val:],
    }


def make_splits(
    data_dir: str,
    ratios: tuple = (0.7, 0.15, 0.15),
    strategy: str = "random",
    valid_file: str = None,
    seed: int = 42,
) -> dict:
    """
    Create train / val / test splits and save to {data_dir}/splits.json.

    Parameters
    ----------
    data_dir : str
        Pipeline output directory.
    ratios : tuple
        (train, val, test) fractions summing to 1 (default 0.7 / 0.15 / 0.15).
    strategy : str
        "random"   — random shuffle.
        "temporal" — patches sorted by acquisition date; earlier → train, later → test.
        "spatial"  — patches grouped by S2 tile code; tiles assigned to splits as units.
    valid_file : str, optional
        Path to valid_patches.txt from dl4eo.qc.validate(); only valid patches used.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    dict  {"train": [...], "val": [...], "test": [...]}
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
    assert strategy in ("random", "temporal", "spatial"), \
        "strategy must be 'random', 'temporal', or 'spatial'"

    stems = _stems_from_dir(data_dir, valid_file)
    rng   = random.Random(seed)
    print(f"[INFO] Splitting {len(stems)} patches — strategy='{strategy}', "
          f"ratios={ratios}, seed={seed}")

    if strategy == "random":
        result = _assign_ratios(stems, ratios, rng)

    elif strategy == "temporal":
        # Parse 8-digit date from filename: S2A_45RXM_20210603_0_L2A_1 → 20210603
        date_re = re.compile(r"_(\d{8})_")
        def _date(stem):
            m = date_re.search(stem)
            return m.group(1) if m else "00000000"

        by_date: dict[str, list] = defaultdict(list)
        for s in stems:
            by_date[_date(s)].append(s)

        sorted_dates = sorted(by_date.keys())
        n_dates = len(sorted_dates)
        n_train_d = round(ratios[0] * n_dates)
        n_val_d   = round(ratios[1] * n_dates)

        train_dates = set(sorted_dates[:n_train_d])
        val_dates   = set(sorted_dates[n_train_d: n_train_d + n_val_d])
        test_dates  = set(sorted_dates[n_train_d + n_val_d:])

        result = {"train": [], "val": [], "test": []}
        for d, patches in by_date.items():
            rng.shuffle(patches)
            if d in train_dates:
                result["train"].extend(patches)
            elif d in val_dates:
                result["val"].extend(patches)
            else:
                result["test"].extend(patches)

    else:  # spatial: split by S2 tile code (e.g. 45RXM)
        tile_re = re.compile(r"S2[AB]_([0-9]{2}[A-Z]{3})_")
        def _tile(stem):
            m = tile_re.search(stem)
            return m.group(1) if m else "UNKNOWN"

        by_tile: dict[str, list] = defaultdict(list)
        for s in stems:
            by_tile[_tile(s)].append(s)

        # Assign whole tiles to splits preserving approximate ratios
        tiles = list(by_tile.keys())
        rng.shuffle(tiles)
        n = len(tiles)
        n_train_t = max(1, round(ratios[0] * n))
        n_val_t   = max(1, round(ratios[1] * n))

        train_tiles = set(tiles[:n_train_t])
        val_tiles   = set(tiles[n_train_t: n_train_t + n_val_t])

        result = {"train": [], "val": [], "test": []}
        for tile, patches in by_tile.items():
            rng.shuffle(patches)
            if tile in train_tiles:
                result["train"].extend(patches)
            elif tile in val_tiles:
                result["val"].extend(patches)
            else:
                result["test"].extend(patches)

    for k, v in result.items():
        print(f"  {k:5s}: {len(v)} patches")

    out_path = os.path.join(data_dir, "splits.json")
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"[✓] Splits saved → {out_path}")
    return result


def load(split_file: str) -> dict:
    """Load splits previously saved by make_splits()."""
    with open(split_file) as fh:
        return json.load(fh)
