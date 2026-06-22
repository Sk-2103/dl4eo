"""
Sentinel-1 RTC download and stacking stage.

Key efficiency improvements:
  - Patches are grouped by acquisition date.
  - STAC search runs once per date (not once per patch).
  - Each SAR granule is downloaded once (atomic rename prevents races).
  - Patch geometry is derived directly from the raster bounds (no per-patch
    shapefile directory needed).
"""

import os
import time
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from shapely.geometry import box, shape as shapely_shape
import geopandas as gpd
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from pystac_client import Client
from planetary_computer import sign
from joblib import Parallel, delayed


_catalog = None


def _get_catalog():
    global _catalog
    if _catalog is None:
        _catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    return _catalog


def _patch_geom_wgs84(patch_path):
    """Return a Shapely box for the patch extent in WGS84."""
    with rasterio.open(patch_path) as src:
        bounds = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return box(*bounds)


def _search_s1_rtc(bounds, start_date, end_date, max_retries=3):
    catalog = _get_catalog()
    for attempt in range(max_retries):
        try:
            search = catalog.search(
                collections=["sentinel-1-rtc"],
                bbox=bounds,
                datetime=f"{start_date}/{end_date}",
                query={"sar:instrument_mode": {"eq": "IW"}},
            )
            return list(search.item_collection())
        except Exception as e:
            print(f"[WARN] STAC search attempt {attempt + 1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                time.sleep(30)
    return []


def _download_asset(href, path, label, max_attempts=3):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    tmp = path + ".tmp"
    for attempt in range(max_attempts):
        try:
            with requests.get(href, stream=True, timeout=(10, 300)) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
            os.replace(tmp, path)
            return True
        except Exception as e:
            print(f"[ERROR] Download {label} attempt {attempt + 1}: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
    return False


def _clip_to_patch(src_path, out_path, ref_crs, ref_transform, ref_h, ref_w):
    """Reproject/resample a SAR band to exactly match an S2 patch grid."""
    with rasterio.open(src_path) as src:
        data = np.empty((ref_h, ref_w), dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )
    meta = {
        "driver": "GTiff", "dtype": "float32", "nodata": 0.0,
        "count": 1, "crs": ref_crs, "transform": ref_transform,
        "width": ref_w, "height": ref_h,
    }
    tmp = out_path + ".tmp"
    with rasterio.open(tmp, "w", **meta) as dst:
        dst.write(data[np.newaxis])
    os.replace(tmp, out_path)


def _process_patch(patch_name, items, target_date,
                   sar_dir, sar_clipped_dir, stacked_dir, stacked_sar_dir):
    """Clip one patch from pre-searched SAR items and stack with S2+DEM."""
    out_stack = os.path.join(stacked_sar_dir, patch_name)
    if os.path.exists(out_stack):
        print(f"[SKIP] Already stacked: {patch_name}")
        return True

    s2_path = os.path.join(stacked_dir, patch_name)
    if not os.path.exists(s2_path):
        print(f"[ERROR] S2+DEM patch missing: {s2_path}")
        return False

    # Derive WGS84 patch geometry directly from the raster
    try:
        patch_geom = _patch_geom_wgs84(s2_path)
    except Exception as e:
        print(f"[ERROR] Cannot read bounds for {patch_name}: {e}")
        return False

    # Find the item whose footprint covers this patch (closest in time wins ties)
    rtc_item = None
    for item in sorted(items, key=lambda it: abs((it.datetime.date() - target_date.date()).days)):
        if shapely_shape(item.geometry).intersects(patch_geom):
            rtc_item = item
            break
    if rtc_item is None:
        rtc_item = min(items, key=lambda it: abs((it.datetime.date() - target_date.date()).days))

    vv = rtc_item.assets.get("vv")
    vh = rtc_item.assets.get("vh")
    hh = rtc_item.assets.get("hh")
    hv = rtc_item.assets.get("hv")

    if vv and vh:
        pol1, pol2, a1, a2 = "VV", "VH", vv, vh
    elif hh and hv:
        pol1, pol2, a1, a2 = "HH", "HV", hh, hv
        print(f"[INFO] Using HH/HV fallback for {rtc_item.id}")
    else:
        print(f"[ERROR] No dual-pol assets in {rtc_item.id}")
        return False

    sar1 = os.path.join(sar_dir, f"{rtc_item.id}_{pol1}.tif")
    sar2 = os.path.join(sar_dir, f"{rtc_item.id}_{pol2}.tif")

    for asset, path, pol in [(a1, sar1, pol1), (a2, sar2, pol2)]:
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            signed = sign(asset.href)
            print(f"[DOWNLOAD] {pol}: {os.path.basename(path)}")
            if not _download_asset(signed, path, pol):
                return False

    with rasterio.open(s2_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_h, ref_w = ref.height, ref.width
        s2_data = ref.read().astype("float32")
        meta = ref.meta.copy()

    shp_stem = os.path.splitext(patch_name)[0]
    clip1 = os.path.join(sar_clipped_dir, f"{shp_stem}_{pol1}.tif")
    clip2 = os.path.join(sar_clipped_dir, f"{shp_stem}_{pol2}.tif")

    try:
        if not os.path.exists(clip1):
            _clip_to_patch(sar1, clip1, ref_crs, ref_transform, ref_h, ref_w)
        if not os.path.exists(clip2):
            _clip_to_patch(sar2, clip2, ref_crs, ref_transform, ref_h, ref_w)
    except Exception as e:
        print(f"[ERROR] SAR clip for {patch_name}: {e}")
        return False

    try:
        with rasterio.open(clip1) as c1, rasterio.open(clip2) as c2:
            d1 = c1.read(1, out_dtype="float32")[np.newaxis]
            d2 = c2.read(1, out_dtype="float32")[np.newaxis]
        stacked = np.concatenate([s2_data, d1, d2], axis=0)
        meta.update(count=stacked.shape[0], dtype="float32")
        with rasterio.open(out_stack, "w", **meta) as dst:
            dst.write(stacked)
        print(f"[✓] Stacked {patch_name}: {stacked.shape[0]} bands")
        return True
    except Exception as e:
        print(f"[ERROR] Final stack for {patch_name}: {e}")
        return False


def run(cfg):
    if cfg.skip_sentinel1:
        print("[SKIP] Sentinel-1 stage disabled (skip_sentinel1=True)")
        return

    print("=" * 80)
    print("[START] Sentinel-1 RTC processing")
    start_time = time.time()

    os.makedirs(cfg.sar_dir, exist_ok=True)
    os.makedirs(cfg.sar_clipped, exist_ok=True)
    os.makedirs(cfg.stacked_with_sar_dir, exist_ok=True)

    # Patches come from the DEM-stacked directory
    stacked_dir = cfg.stacked_dir
    patch_files = sorted(f for f in os.listdir(stacked_dir) if f.endswith(".tif"))
    print(f"[INFO] {len(patch_files)} patch(es) to process")

    # Group by 8-digit acquisition date in filename
    date_groups: dict[str, list[str]] = defaultdict(list)
    for pf in patch_files:
        stem = os.path.splitext(pf)[0]
        try:
            date_str = next(s for s in stem.split("_") if s.isdigit() and len(s) == 8)
            date_groups[date_str].append(pf)
        except StopIteration:
            print(f"[WARN] No date in filename: {pf}")

    print(f"[INFO] {len(date_groups)} unique acquisition date(s)")

    total_ok = total_fail = 0

    for date_str, patch_names in sorted(date_groups.items()):
        target_date = datetime.strptime(date_str, "%Y%m%d")
        delta = getattr(cfg, "sar_days_delta", 5)
        start_d = (target_date - timedelta(days=delta)).strftime("%Y-%m-%d")
        end_d   = (target_date + timedelta(days=delta)).strftime("%Y-%m-%d")

        # Compute WGS84 bounding box union across all patches in this date group
        all_geoms = []
        for pn in patch_names:
            patch_path = os.path.join(stacked_dir, pn)
            if os.path.exists(patch_path):
                try:
                    all_geoms.append(_patch_geom_wgs84(patch_path))
                except Exception:
                    pass
        if not all_geoms:
            continue

        from shapely.ops import unary_union
        group_union = unary_union(all_geoms)
        search_bounds = group_union.bounds  # (minx, miny, maxx, maxy)

        print(f"[STAC] Searching date {date_str} ({len(patch_names)} patches)")
        items = _search_s1_rtc(search_bounds, start_d, end_d)
        if not items:
            print(f"[FAIL] No S1 RTC items for {date_str}")
            total_fail += len(patch_names)
            continue

        results = Parallel(n_jobs=max(1, cfg.n_jobs // 2), prefer="threads")(
            delayed(_process_patch)(
                pn, items, target_date,
                cfg.sar_dir, cfg.sar_clipped,
                stacked_dir, cfg.stacked_with_sar_dir,
            )
            for pn in patch_names
        )
        total_ok   += sum(1 for r in results if r)
        total_fail += sum(1 for r in results if not r)

    elapsed = time.time() - start_time
    print(f"\n[DONE] Sentinel-1: {total_ok} succeeded, {total_fail} failed "
          f"in {elapsed:.1f}s")
    if total_fail:
        print(f"[INFO] Check {os.path.join(cfg.base_dir, 'failed_sar.txt')} for details")
    print("=" * 80)
