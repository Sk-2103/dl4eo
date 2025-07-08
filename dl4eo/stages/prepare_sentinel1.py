def run(cfg):
    import time
    import os
    import rasterio
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import box
    from rasterio.mask import mask
    from rasterio.warp import reproject, Resampling
    from joblib import Parallel, delayed
    import traceback
    from pystac_client import Client
    from planetary_computer import sign
    import requests
    from datetime import datetime, timedelta
    import multiprocessing
    import shutil

    print("=" * 80)
    print("[START] Sentinel-1 RTC dual-pol + Sentinel-2 stacking pipeline with HH/HV fallback...")
    start_time = time.time()

    # === Convert shapefiles to WGS84 ===
    input_folder = cfg.shapefile_each
    output_folder = cfg.stacked_sample_wgs84
    os.makedirs(output_folder, exist_ok=True)

    shapefiles = [f for f in os.listdir(input_folder) if f.endswith(".shp")]
    print(f"[INFO] Found {len(shapefiles)} shapefiles to convert to WGS84...")

    def convert_to_wgs84(file):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        if os.path.exists(output_path):
            print(f"[SKIP] WGS84 shapefile exists: {file}")
            return
        try:
            gdf = gpd.read_file(input_path)
            gdf.to_crs(epsg=4326).to_file(output_path)
            print(f"[✓] Converted: {file}")
        except Exception as e:
            print(f"[ERROR] WGS84 conversion failed for {file}: {e}")

    Parallel(n_jobs=cfg.n_jobs)(delayed(convert_to_wgs84)(f) for f in shapefiles)

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    shapefile_folder = cfg.stacked_sample_wgs84
    os.makedirs(cfg.sar_dir, exist_ok=True)
    os.makedirs(cfg.sar_clipped, exist_ok=True)
    os.makedirs(cfg.stacked_with_sar_dir, exist_ok=True)

    remaining_shapefiles = [os.path.join(shapefile_folder, f) for f in os.listdir(shapefile_folder) if f.endswith(".shp")]
    processed_shapefiles = set()
    failed_shapefiles = set()

    def robust_search(catalog, union_geom, start_date, end_date, max_retries=3, delay=30):
        for attempt in range(max_retries):
            try:
                search = catalog.search(
                    collections=["sentinel-1-rtc"],
                    bbox=union_geom.bounds,
                    datetime=f"{start_date}/{end_date}",
                    query={"sar:instrument_mode": {"eq": "IW"}}
                )
                return list(search.items())
            except Exception as e:
                print(f"[WARN] STAC request failed (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
                else:
                    return None

    def download_with_retry(href, path, pol, max_attempts=3):
        for attempt in range(max_attempts):
            try:
                print(f"[DOWNLOAD] {pol} → {path} (attempt {attempt+1})")
                with requests.get(href, stream=True, timeout=(10, 120)) as r:
                    r.raise_for_status()
                    with open(path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                print(f"[✓] Downloaded {pol}")
                return True
            except Exception as e:
                print(f"[ERROR] Download failed for {pol}: {e}")
                if attempt < max_attempts - 1:
                    wait = 2 ** attempt
                    print(f"[RETRY] Waiting {wait}s before retry...")
                    time.sleep(wait)
                else:
                    return False

    def process_shapefile(shapefile_path):
        shapefile_name = os.path.splitext(os.path.basename(shapefile_path))[0]
        try:
            date_str = [s for s in shapefile_name.split('_') if s.isdigit() and len(s) == 8][0]
            target_date = datetime.strptime(date_str, '%Y%m%d')
        except Exception as e:
            print(f"[ERROR] Failed to extract date: {e}")
            failed_shapefiles.add(shapefile_name)
            return

        try:
            gdf = gpd.read_file(shapefile_path)
            gdf = gdf[gdf.is_valid & ~gdf.is_empty]
            union_geom = gdf.geometry.unary_union
        except Exception as e:
            print(f"[ERROR] Failed to load geometry: {e}")
            failed_shapefiles.add(shapefile_name)
            return

        start_date = (target_date - timedelta(days=10)).strftime("%Y-%m-%d")
        end_date = (target_date + timedelta(days=10)).strftime("%Y-%m-%d")
        print(f"[INFO] Searching RTC in range: {start_date} to {end_date}")

        items = robust_search(catalog, union_geom, start_date, end_date)
        if not items:
            print(f"[FAIL] No RTC items retrieved.")
            failed_shapefiles.add(shapefile_name)
            return

        rtc_item = min(items, key=lambda item: abs((item.datetime.date() - target_date.date()).days))

        asset_vv = rtc_item.assets.get("vv")
        asset_vh = rtc_item.assets.get("vh")
        asset_hh = rtc_item.assets.get("hh")
        asset_hv = rtc_item.assets.get("hv")

        if asset_vv and asset_vh:
            pol1, pol2 = "VV", "VH"
            asset1, asset2 = asset_vv, asset_vh
        elif asset_hh and asset_hv:
            pol1, pol2 = "HH", "HV"
            asset1, asset2 = asset_hh, asset_hv
            print(f"[INFO] Using HH/HV instead of VV/VH for: {rtc_item.id}")
        else:
            print(f"[ERROR] RTC item missing both VV/VH and HH/HV.")
            failed_shapefiles.add(shapefile_name)
            return

        href_1 = sign(asset1.href)
        href_2 = sign(asset2.href)

        rtc_path_1 = os.path.join(cfg.sar_dir, f"{rtc_item.id}_{pol1}.tif")
        rtc_path_2 = os.path.join(cfg.sar_dir, f"{rtc_item.id}_{pol2}.tif")

        for href, path, pol in [(href_1, rtc_path_1, pol1), (href_2, rtc_path_2, pol2)]:
            if not os.path.exists(path):
                if not download_with_retry(href, path, pol):
                    failed_shapefiles.add(shapefile_name)
                    return

        s2_path = os.path.join(cfg.s2_images, f"{shapefile_name}.tif")
        if not os.path.exists(s2_path):
            print(f"[ERROR] S2 image not found: {s2_path}")
            failed_shapefiles.add(shapefile_name)
            return

        with rasterio.open(s2_path) as ref:
            ref_crs, ref_transform = ref.crs, ref.transform
            ref_width, ref_height = ref.width, ref.height
            bounds = ref.bounds

        def clip_and_reproject(src_path, out_path, pol):
            print(f"[PROCESSING] Clip + Reproject {pol}")
            with rasterio.open(src_path) as src:
                data = np.empty((src.count, ref_height, ref_width), dtype=src.dtypes[0])
                meta = src.meta.copy()
                meta.update({
                    "crs": ref_crs, "transform": ref_transform,
                    "width": ref_width, "height": ref_height
                })

                for i in range(src.count):
                    reproject(
                        source=rasterio.band(src, i + 1),
                        destination=data[i],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear
                    )

            temp_path = out_path.replace(".tif", "_temp.tif")
            with rasterio.open(temp_path, "w", **meta) as dst:
                dst.write(data)

            geom = [box(*bounds)]
            with rasterio.open(temp_path) as src:
                out_image, out_transform = mask(src, geom, crop=True, nodata=0)
                out_meta = meta.copy()
                out_meta.update({
                    "height": out_image.shape[1], "width": out_image.shape[2],
                    "transform": out_transform, "nodata": 0
                })
                with rasterio.open(out_path, "w", **out_meta) as dst:
                    dst.write(out_image)
            os.remove(temp_path)
            print(f"[✓] Clipped: {out_path}")

        clip_1 = os.path.join(cfg.sar_clipped, f"{shapefile_name}_{pol1}.tif")
        clip_2 = os.path.join(cfg.sar_clipped, f"{shapefile_name}_{pol2}.tif")
        clip_and_reproject(rtc_path_1, clip_1, pol1)
        clip_and_reproject(rtc_path_2, clip_2, pol2)

        stacked_path = os.path.join(cfg.stacked_dir, f"{shapefile_name}.tif")
        out_stack = os.path.join(cfg.stacked_with_sar_dir, f"{shapefile_name}.tif")

        if not os.path.exists(stacked_path):
            print(f"[ERROR] Stacked S2 image missing: {stacked_path}")
            failed_shapefiles.add(shapefile_name)
            return

        if os.path.exists(out_stack):
            print(f"[SKIP] Already stacked: {shapefile_name}")
            processed_shapefiles.add(shapefile_name)
            return

        with rasterio.open(stacked_path) as s2_src, \
             rasterio.open(clip_1) as src1, \
             rasterio.open(clip_2) as src2:

            s2_data = s2_src.read()
            data1 = src1.read(1)[np.newaxis, ...]
            data2 = src2.read(1)[np.newaxis, ...]
            stacked_data = np.concatenate((s2_data, data1, data2), axis=0)

            out_meta = s2_src.meta.copy()
            out_meta.update({"count": stacked_data.shape[0]})

            with rasterio.open(out_stack, "w", **out_meta) as dst:
                dst.write(stacked_data)
            print(f"[✓] Stacked {shapefile_name}: {stacked_data.shape[0]} bands")

        processed_shapefiles.add(shapefile_name)

    def run_with_timeout(func, args=(), timeout=1500):
        proc = multiprocessing.Process(target=func, args=args)
        proc.start()
        proc.join(timeout)
        if proc.is_alive():
            print(f"[TIMEOUT] Killing process for: {os.path.basename(args[0])}")
            proc.terminate()
            proc.join()
            shapefile_name = os.path.splitext(os.path.basename(args[0]))[0]
            failed_shapefiles.add(shapefile_name)

    while remaining_shapefiles:
        for shapefile_path in remaining_shapefiles:
            shapefile_name = os.path.splitext(os.path.basename(shapefile_path))[0]
            if shapefile_name in processed_shapefiles or shapefile_name in failed_shapefiles:
                continue
            print(f"\n[PROCESSING] {shapefile_name}")
            run_with_timeout(process_shapefile, args=(shapefile_path,), timeout=1500)

        remaining_shapefiles = [
            shp for shp in remaining_shapefiles
            if os.path.splitext(os.path.basename(shp))[0] not in processed_shapefiles
            and os.path.splitext(os.path.basename(shp))[0] not in failed_shapefiles
        ]

    print("\n" + "=" * 80)
    print(f"[DONE] Completed in {time.time() - start_time:.2f} seconds")
    print(f"[✓] Processed: {len(processed_shapefiles)}")
    print(f"[✗] Failed: {len(failed_shapefiles)}")

    if failed_shapefiles:
        failed_txt = os.path.join(cfg.base_dir, "failed_sar_clips.txt")
        with open(failed_txt, "w") as f:
            for name in sorted(failed_shapefiles):
                f.write(name + "\n")
        print(f"[INFO] Wrote failed list to: {failed_txt}")

    print("=" * 80)
