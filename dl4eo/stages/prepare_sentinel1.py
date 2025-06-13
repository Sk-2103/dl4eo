def run(cfg):
    import os
    import time
    import rasterio
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import box, shape
    from rasterio.mask import mask
    from rasterio.warp import reproject, Resampling
    from joblib import Parallel, delayed
    import traceback
    from pystac_client import Client
    from planetary_computer import sign
    import requests
    from datetime import datetime, timedelta
    
    print("=" * 80)
    print("[START] Sentinel-1 RTC VV+VH + Sentinel-2 stacking pipeline...")
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
    
    # === Prepare STAC Catalog and folders ===
    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    shapefile_folder = cfg.stacked_sample_wgs84
    os.makedirs(cfg.sar_dir, exist_ok=True)
    os.makedirs(cfg.sar_clipped, exist_ok=True)
    os.makedirs(cfg.stacked_with_sar_dir, exist_ok=True)
    
    remaining_shapefiles = [os.path.join(shapefile_folder, f) for f in os.listdir(shapefile_folder) if f.endswith(".shp")]
    processed_shapefiles = set()
    failed_shapefiles = set()
    
    # === Retry wrapper for STAC search ===
    def robust_search(catalog, union_geom, start_date, end_date, max_retries=3, delay=10):
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
    
    while remaining_shapefiles:
        for shapefile_path in remaining_shapefiles:
            shapefile_name = os.path.splitext(os.path.basename(shapefile_path))[0]
            if shapefile_name in processed_shapefiles or shapefile_name in failed_shapefiles:
                continue
    
            print(f"\n[PROCESSING] {shapefile_name}")
    
            try:
                date_str = [s for s in shapefile_name.split('_') if s.isdigit() and len(s) == 8][0]
                target_date = datetime.strptime(date_str, '%Y%m%d')
            except Exception as e:
                print(f"[ERROR] Failed to extract date: {e}")
                failed_shapefiles.add(shapefile_name)
                continue
    
            try:
                gdf = gpd.read_file(shapefile_path)
                gdf = gdf[gdf.is_valid & ~gdf.is_empty]
                union_geom = gdf.geometry.unary_union
            except Exception as e:
                print(f"[ERROR] Failed to load geometry: {e}")
                failed_shapefiles.add(shapefile_name)
                continue
    
            start_date = (target_date - timedelta(days=10)).strftime("%Y-%m-%d")
            end_date = (target_date + timedelta(days=10)).strftime("%Y-%m-%d")
            print(f"[INFO] Searching RTC in range: {start_date} to {end_date}")
    
            items = robust_search(catalog, union_geom, start_date, end_date)
            if not items:
                print(f"[FAIL] No RTC items retrieved.")
                failed_shapefiles.add(shapefile_name)
                continue
    
            rtc_item = min(items, key=lambda item: abs((item.datetime.date() - target_date.date()).days))
            asset_vv = rtc_item.assets.get("vv")
            asset_vh = rtc_item.assets.get("vh")
    
            if not asset_vv or not asset_vh:
                print(f"[ERROR] RTC item missing VV or VH.")
                failed_shapefiles.add(shapefile_name)
                continue
    
            href_vv = sign(asset_vv.href)
            href_vh = sign(asset_vh.href)
    
            rtc_vv_path = os.path.join(cfg.sar_dir, f"{rtc_item.id}_VV.tif")
            rtc_vh_path = os.path.join(cfg.sar_dir, f"{rtc_item.id}_VH.tif")
    
            for href, path, pol in [(href_vv, rtc_vv_path, "VV"), (href_vh, rtc_vh_path, "VH")]:
                if not os.path.exists(path):
                    try:
                        print(f"[DOWNLOAD] {pol} → {path}")
                        with requests.get(href, stream=True) as r:
                            r.raise_for_status()
                            with open(path, 'wb') as f:
                                for chunk in r.iter_content(chunk_size=8192):
                                    f.write(chunk)
                        print(f"[✓] Downloaded {pol}")
                    except Exception as e:
                        print(f"[ERROR] Download failed for {pol}: {e}")
                        failed_shapefiles.add(shapefile_name)
                        continue
    
            s2_path = os.path.join(cfg.s2_images, f"{shapefile_name}.tif")
            if not os.path.exists(s2_path):
                print(f"[ERROR] S2 image not found: {s2_path}")
                failed_shapefiles.add(shapefile_name)
                continue
    
            try:
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
    
                vv_clip = os.path.join(cfg.sar_clipped, f"{shapefile_name}_VV.tif")
                vh_clip = os.path.join(cfg.sar_clipped, f"{shapefile_name}_VH.tif")
                clip_and_reproject(rtc_vv_path, vv_clip, "VV")
                clip_and_reproject(rtc_vh_path, vh_clip, "VH")
    
                stacked_path = os.path.join(cfg.stacked_dir, f"{shapefile_name}.tif")
                out_stack = os.path.join(cfg.stacked_with_sar_dir, f"{shapefile_name}.tif")
    
                if not os.path.exists(stacked_path):
                    print(f"[ERROR] Stacked S2 image missing: {stacked_path}")
                    failed_shapefiles.add(shapefile_name)
                    continue
    
                if os.path.exists(out_stack):
                    print(f"[SKIP] Already stacked: {shapefile_name}")
                    processed_shapefiles.add(shapefile_name)
                    continue
    
                with rasterio.open(stacked_path) as s2_src, \
                     rasterio.open(vv_clip) as vv_src, \
                     rasterio.open(vh_clip) as vh_src:
    
                    s2_data = s2_src.read()
                    vv_data = vv_src.read(1)[np.newaxis, ...]
                    vh_data = vh_src.read(1)[np.newaxis, ...]
                    stacked_data = np.concatenate((s2_data, vv_data, vh_data), axis=0)
    
                    out_meta = s2_src.meta.copy()
                    out_meta.update({"count": stacked_data.shape[0]})
    
                    with rasterio.open(out_stack, "w", **out_meta) as dst:
                        dst.write(stacked_data)
                    print(f"[✓] Stacked {shapefile_name}: {stacked_data.shape[0]} bands")
    
                processed_shapefiles.add(shapefile_name)
    
            except Exception as e:
                print(f"[ERROR] Failed to process {shapefile_name}: {e}")
                traceback.print_exc()
                failed_shapefiles.add(shapefile_name)
    
        remaining_shapefiles = [
            shp for shp in remaining_shapefiles
            if os.path.splitext(os.path.basename(shp))[0] not in processed_shapefiles
            and os.path.splitext(os.path.basename(shp))[0] not in failed_shapefiles
        ]
    
    # === Summary ===
    print("\n" + "=" * 80)
    print(f"[DONE] Completed in {time.time() - start_time:.2f} seconds")
    print(f"[✓] Processed: {len(processed_shapefiles)}")
    print(f"[✗] Failed: {len(failed_shapefiles)}")
    
    if failed_shapefiles:
        with open("failed_sar_clips.txt", "w") as f:
            for name in sorted(failed_shapefiles):
                f.write(name + "\n")
        print("[INFO] Wrote failed list to: failed_sar_clips.txt")
    
    print("=" * 80)