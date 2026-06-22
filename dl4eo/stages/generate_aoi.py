def run(cfg):
    import time
    import os
    import pandas as pd
    import geopandas as gpd
    import rasterio
    import rasterio.windows as rio_win
    import numpy as np
    from shapely.geometry import box
    from shapely.strtree import STRtree
    from rasterio.windows import from_bounds as win_from_bounds
    from rasterio.errors import RasterioError
    from joblib import Parallel, delayed

    print("=" * 60)
    print("[START] Generating patch AOI boxes and clipping images")
    start_time = time.time()

    raster_folder = cfg.s2_stack
    label_shapefile = cfg.lake_shp_path
    aoi_boxes_dir = cfg.aoi_boxes
    patches_dir = cfg.s2_images   # clipped patch TIFs written here
    os.makedirs(aoi_boxes_dir, exist_ok=True)
    os.makedirs(patches_dir, exist_ok=True)

    lake_gdf = gpd.read_file(label_shapefile)

    # --- Step 1: Generate tiling boxes for each stacked raster ---
    def generate_boxes(raster_path):
        raster_name = os.path.splitext(os.path.basename(raster_path))[0]
        out_path = os.path.join(aoi_boxes_dir, f"{raster_name}_aoi_boxes.shp")
        if os.path.exists(out_path):
            print(f"[SKIP] AOI boxes exist: {raster_name}")
            return

        try:
            with rasterio.open(raster_path) as src:
                transform = src.transform
                crs = src.crs
                w, h = src.width, src.height
                minx, maxy = transform * (0, 0)
                maxx, miny = transform * (w, h)
                raster_bounds = box(minx, miny, maxx, maxy)

            lake_proj = lake_gdf.to_crs(crs)
            lake_geoms = lake_proj.geometry
            lake_geoms = lake_geoms[~lake_geoms.is_empty & lake_geoms.is_valid]
            if lake_geoms.empty:
                print(f"[SKIP] No valid label geometries after reprojection: {raster_name}")
                return

            lake_list = list(lake_geoms)
            if not any(g.intersects(raster_bounds) for g in lake_list):
                print(f"[SKIP] No labels intersect raster: {raster_name}")
                return

            lake_idx = STRtree(lake_list)
            box_m = cfg.box_size_m
            stride_m = cfg.stride_m

            # --- Restrict tiling to user's AOI polygon ---
            # Exclude feature_shapefile so lake/label files don't pollute the AOI
            feature_shp_name = os.path.basename(cfg.feature_shapefile)
            aoi_shps = [
                os.path.join(cfg.aoi_shapefile_dir, f)
                for f in os.listdir(cfg.aoi_shapefile_dir)
                if f.endswith(".shp") and f != feature_shp_name
            ]
            if aoi_shps:
                frames = [gpd.read_file(p) for p in aoi_shps]
                aoi_gdf = gpd.GeoDataFrame(
                    pd.concat(frames, ignore_index=True), crs=frames[0].crs
                )
                aoi_union = aoi_gdf.to_crs(crs).geometry.unary_union
                tile_region = aoi_union.intersection(raster_bounds)
                if tile_region.is_empty:
                    print(f"[SKIP] AOI does not overlap raster: {raster_name}")
                    return
            else:
                aoi_union = raster_bounds
                tile_region = raster_bounds

            tx_min, ty_min, tx_max, ty_max = tile_region.bounds

            boxes = []
            x = tx_min
            while x + box_m <= tx_max + 1e-3:
                y = ty_min
                while y + box_m <= ty_max + 1e-3:
                    candidate = box(x, y, x + box_m, y + box_m)
                    if candidate.intersects(aoi_union):
                        hits = lake_idx.query(candidate)
                        if any(lake_list[i].intersects(candidate) for i in hits):
                            boxes.append(candidate)
                    y += stride_m
                x += stride_m

            if not boxes:
                print(f"[SKIP] No label-intersecting boxes found within AOI: {raster_name}")
                return

            gpd.GeoDataFrame(geometry=boxes, crs=crs).to_file(out_path)
            print(f"[✓] {len(boxes)} AOI boxes → {raster_name}")
        except Exception as e:
            print(f"[ERROR] generate_boxes({raster_name}): {e}")

    rasters = [
        os.path.join(raster_folder, f)
        for f in os.listdir(raster_folder) if f.endswith(".tif")
    ]
    Parallel(n_jobs=cfg.n_jobs)(delayed(generate_boxes)(r) for r in rasters)
    print(f"[INFO] AOI box generation done ({len(rasters)} rasters).")

    # --- Step 2: Clip stacked rasters to each AOI box → individual patches ---
    # Uses windowed reads: only touches the ~256×256 px region, not the full tile.
    def clip_image(image_file):
        base_name = os.path.splitext(image_file)[0]
        shp_path = os.path.join(aoi_boxes_dir, f"{base_name}_aoi_boxes.shp")
        image_path = os.path.join(raster_folder, image_file)

        if not os.path.exists(shp_path):
            print(f"[SKIP] No AOI boxes shapefile for {image_file}")
            return

        try:
            gdf = gpd.read_file(shp_path)
        except Exception as e:
            print(f"[ERROR] Reading {shp_path}: {e}")
            return

        valid_rows = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
        if valid_rows.empty:
            return

        try:
            with rasterio.open(image_path) as src:
                full_win = rio_win.Window(0, 0, src.width, src.height)
                for idx, (_, row) in enumerate(valid_rows.iterrows(), 1):
                    out_file = os.path.join(patches_dir, f"{base_name}_{idx}.tif")
                    if os.path.exists(out_file):
                        continue
                    try:
                        geom = row.geometry
                        win = win_from_bounds(*geom.bounds, transform=src.transform)
                        win = win.round_offsets().round_lengths()
                        win = win.intersection(full_win)
                        if win.width <= 0 or win.height <= 0:
                            continue
                        data = src.read(window=win)
                        if src.nodata is not None and np.all(data == src.nodata):
                            continue
                        win_transform = src.window_transform(win)
                        meta = src.meta.copy()
                        meta.update(
                            height=data.shape[1],
                            width=data.shape[2],
                            transform=win_transform,
                        )
                        with rasterio.open(out_file, "w", **meta) as dst:
                            dst.write(data)
                        print(f"[✓] Patch: {os.path.basename(out_file)}")
                    except (RasterioError, ValueError) as e:
                        print(f"[SKIP] Clip error in {image_file} box {idx}: {e}")
        except Exception as e:
            print(f"[ERROR] clip_image({image_file}): {e}")

    image_files = sorted(f for f in os.listdir(raster_folder) if f.endswith(".tif"))
    Parallel(n_jobs=cfg.n_jobs)(delayed(clip_image)(f) for f in image_files)
    print("[INFO] Image clipping complete.")

    elapsed = time.time() - start_time
    print(f"[DONE] AOI generation complete in {elapsed:.1f}s")
    print("=" * 60)
