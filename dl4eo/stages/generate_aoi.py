def run(cfg):
    import time
    import os
    import geopandas as gpd
    import rasterio
    from shapely.geometry import box, mapping
    from shapely.strtree import STRtree
    from rasterio.mask import mask
    from rasterio.errors import RasterioError
    from joblib import Parallel, delayed
    
    print("="*60)
    print(f"[START] {__file__} running...")
    start_time = time.time()
    
    # === Paths from config ===
    raster_folder = cfg.s2_stack
    lake_shapefile = cfg.lake_shp_path
    output_shapefile_dir = cfg.aoi_boxes
    image_folder = cfg.s2_stack
    shapefile_folder = cfg.aoi_boxes
    output_folder = cfg.s2_images
    os.makedirs(output_shapefile_dir, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)
    
    # === AOI Box Generation ===
    cfg.box_size_m = 2560  # 256 pixels at 10m resolution
    lake_gdf = gpd.read_file(lake_shapefile)
    
    def generate_boxes_for_raster(raster_path):
        raster_name = os.path.splitext(os.path.basename(raster_path))[0]
        print(f"[INFO] Processing: {raster_name}")
    
        try:
            with rasterio.open(raster_path) as src:
                transform = src.transform
                crs = src.crs
                width = src.width
                height = src.height
    
                minx, maxy = transform * (0, 0)
                maxx, miny = transform * (width, height)
                raster_bounds = box(minx, miny, maxx, maxy)
    
            lake_gdf_proj = lake_gdf.to_crs(crs)
            lake_geoms = lake_gdf_proj.geometry
            lake_geoms = lake_geoms[~lake_geoms.is_empty & lake_geoms.is_valid]
            lake_geom_list = list(lake_geoms)
    
            if not any(geom.intersects(raster_bounds) for geom in lake_geom_list):
                print(f"[SKIP] No lakes intersect raster extent: {raster_name}")
                return
    
            lake_index = STRtree(lake_geom_list)
    
            x_steps = int((maxx - minx) // cfg.box_size_m)
            y_steps = int((maxy - miny) // cfg.box_size_m)
    
            boxes = []
            for i in range(x_steps + 1):
                for j in range(y_steps + 1):
                    x_min = minx + i * cfg.box_size_m
                    y_min = miny + j * cfg.box_size_m
                    x_max = x_min + cfg.box_size_m
                    y_max = y_min + cfg.box_size_m
    
                    candidate_box = box(x_min, y_min, x_max, y_max)
                    idx_matches = lake_index.query(candidate_box)
    
                    if any(lake_geom_list[idx].intersects(candidate_box) for idx in idx_matches):
                        boxes.append(candidate_box)
    
            if not boxes:
                print(f"[SKIP] No lake-intersecting AOI boxes: {raster_name}")
                return
    
            output_gdf = gpd.GeoDataFrame(geometry=boxes, crs=crs)
            output_path = os.path.join(output_shapefile_dir, f"{raster_name}_aoi_boxes.shp")
            output_gdf.to_file(output_path)
            print(f"[SAVED] AOI boxes saved: {output_path}")
    
        except Exception as e:
            print(f"[ERROR] Failed on {raster_name}: {e}")
    
    raster_files = [os.path.join(raster_folder, f) for f in os.listdir(raster_folder) if f.endswith(".tif")]
    Parallel(n_jobs=cfg.n_jobs)(delayed(generate_boxes_for_raster)(raster_path) for raster_path in raster_files)
    print("✅ AOI generation completed for all rasters.")
    
    # === AOI-Based Clipping ===
    def process_image(image_file):
        image_path = os.path.join(image_folder, image_file)
        base_name = os.path.splitext(image_file)[0]
        shapefile_path = os.path.join(shapefile_folder, f"{base_name}_aoi_boxes.shp")
    
        if not os.path.exists(shapefile_path) or os.path.getsize(shapefile_path) < 1000:
            print(f"[SKIP] Shapefile not found or too small for {image_file}")
            return
    
        print(f"[INFO] Processing: {image_file} with {shapefile_path}")
        try:
            gdf = gpd.read_file(shapefile_path)
            if gdf.empty or not gdf.geometry.is_valid.all():
                print(f"[SKIP] Invalid or empty geometries in {shapefile_path}")
                return
    
            with rasterio.open(image_path) as src:
                nodata_value = src.nodata
                local_index = 1
    
                for _, row in gdf.iterrows():
                    try:
                        clipped_image, clipped_transform = mask(src, [mapping(row.geometry)], crop=True)
    
                        if nodata_value is not None and (clipped_image == nodata_value).all():
                            print(f"[SKIP] Clipped region is all nodata in {image_file}")
                            continue
    
                        out_meta = src.meta.copy()
                        out_meta.update({
                            "driver": "GTiff",
                            "height": clipped_image.shape[1],
                            "width": clipped_image.shape[2],
                            "transform": clipped_transform
                        })
    
                        output_filename = f"{base_name}_{local_index}.tif"
                        output_path = os.path.join(output_folder, output_filename)
    
                        with rasterio.open(output_path, "w", **out_meta) as dest:
                            dest.write(clipped_image)
    
                        print(f"[SAVED] {output_filename}")
                        local_index += 1
    
                    except (RasterioError, ValueError) as e:
                        print(f"[SKIP] Error clipping shape in {image_file}: {e}")
                        continue
    
        except Exception as e:
            print(f"[ERROR] Failed on {image_file}: {e}")
    
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".tif")])
    Parallel(n_jobs=cfg.n_jobs)(delayed(process_image)(img_file) for img_file in image_files)
    print("✅ All clipping completed.")
    
    end_time = time.time()
    print(f"[DONE] {__file__} completed in {end_time - start_time:.2f} seconds")
    print("="*60)
    