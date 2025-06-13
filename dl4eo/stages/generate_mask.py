def run(cfg):
    import time
    
    print("="*60)
    print(f"[START] {__file__} running...")
    start_time = time.time()
    
    # === STEP 4: Rasterize lake shapefile to match Sentinel-2 images ===
    import os
    import numpy as np
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    from shapely.geometry import box
    from joblib import Parallel, delayed
    import matplotlib.pyplot as plt
    import fiona
    
    print("[INFO] Rasterizing lake shapefile into binary cfg.masks...")
    os.makedirs(cfg.masks, exist_ok=True)
    
    gdf_all = gpd.read_file(cfg.lake_shp_path)
    if 'id' not in gdf_all.columns:
        gdf_all = gdf_all[[gdf_all.geometry.name]].copy()
        gdf_all['id'] = 1
        gdf_all.to_file(cfg.lake_shp_path)
        print(f"[INFO] Updated shapefile saved with 'id': {cfg.lake_shp_path}")
    
    # === EXTRA: Inspect Shapefile Metadata ===
    print("[INFO] Inspecting lake shapefile...")
    if gdf_all.empty:
        print("❌ Error: could not load dataset or it's empty")
    else:
        print("✅ Shapefile loaded successfully!\n")
    
    with fiona.open(cfg.lake_shp_path) as src:
        print(f"Dataset driver is: {src.driver}\n")
    
    try:
        print(f"Layer projection (Proj4): {gdf_all.crs.to_proj4()}\n")
    except:
        print("Layer projection is undefined.\n")
    
    geom_types = gdf_all.geom_type.unique()
    print(f"Layer geometry types: {geom_types}\n")
    print(f"Layer has {len(gdf_all)} features\n")
    print(f"Layer has {len(gdf_all.columns) - 1} fields (excluding geometry):")
    for col in gdf_all.columns:
        if col != 'geometry':
            print(f"\t{col} - {gdf_all[col].dtype}")
    
    gdf_all.head(10).plot(figsize=(8, 6), edgecolor='black', facecolor='skyblue')
    plt.title("Sample of Glacial Lakes")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()
    # plt.show()
    
    # === RASTERIZE ===
    def rasterize_mask(filename):
        if not filename.endswith('.tif'):
            return
        raster_path = os.path.join(cfg.stacked_dir, filename)
        output_path = os.path.join(cfg.masks, filename)
    
        with rasterio.open(raster_path) as src:
            width, height = src.width, src.height
            transform = src.transform
            crs = src.crs
            bounds = src.bounds
    
        raster_bbox = box(*bounds)
        gdf_raster = gdf_all.to_crs(crs)
        gdf_clip = gdf_raster[gdf_raster.geometry.intersects(raster_bbox)]
    
        meta = {
            'driver': 'GTiff',
            'height': height,
            'width': width,
            'count': 1,
            'dtype': 'uint8',
            'crs': crs,
            'transform': transform
        }
    
        if gdf_clip.empty:
            with rasterio.open(output_path, 'w', **meta) as dst:
                dst.write(np.zeros((1, height, width), dtype='uint8'))
            print(f"[SKIP] No intersection: {filename}")
            return
    
        shapes = [(geom, int(val)) for geom, val in zip(gdf_clip.geometry, gdf_clip['id'])]
    
        mask_arr = rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype='uint8',
            all_touched=True
        )
    
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(mask_arr, 1)
    
        print(f"[✓] Mask created: {filename}")
    
    # Process all files in parallel
    stacked_files = os.listdir(cfg.stacked_dir)
    Parallel(n_jobs=cfg.n_jobs)(delayed(rasterize_mask)(f) for f in stacked_files)
    
    end_time = time.time()
    print(f"[DONE] {__file__} completed in {end_time - start_time:.2f} seconds")
    print("="*60)
    