def run(cfg):
    # === START OF SCRIPT ===
    import time
    import os
    import rasterio
    from rasterio.merge import merge
    import numpy as np
    import geopandas as gpd
    from shapely.geometry import box
    from rasterio.mask import mask
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from pathlib import Path
    from subprocess import call
    from joblib import Parallel, delayed
    import shutil
    import glob
    
    print("="*60)
    print(f"[START] {__file__} running...")
    start_time = time.time()
    
    # === Config Paths ===
    input_folder = cfg.s2_images
    output_shapefile_each = cfg.shapefile_each
    cfg.shapefile_dir = output_shapefile_each
    os.makedirs(output_shapefile_each, exist_ok=True)
    
    # === Step 1: Convert image extent to shapefile ===
    def raster_extent_to_vector(raster_path, output_path):
        if os.path.exists(output_path):
            print(f"[SKIP] Shapefile exists: {output_path}")
            return
        with rasterio.open(raster_path) as src:
            bounds = src.bounds
            geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            gdf = gpd.GeoDataFrame({"filename": [os.path.basename(raster_path)], "geometry": [geom]}, crs=src.crs)
            gdf.to_file(output_path, driver="ESRI Shapefile")
            print(f"Saved: {output_path}")
    
    for file in os.listdir(input_folder):
        if file.endswith(".tif"):
            raster_path = os.path.join(input_folder, file)
            image_name = os.path.splitext(file)[0]
            output_path = os.path.join(output_shapefile_each, f"{image_name}.shp")
            try:
                raster_extent_to_vector(raster_path, output_path)
            except Exception as e:
                print(f"Error processing {file}: {e}")
    
    # === Step 2: Download DEMs (copied from cache if available) ===
    destination_crs = "EPSG:4326"
    dem_base_path = cfg.dem_dir
    download_base_path = os.path.join(dem_base_path, "dems")
    cache_folder = os.path.join(dem_base_path, "cache")
    os.makedirs(cache_folder, exist_ok=True)
    
    
    def s3Path(lat, lon):
        lonSign = {1: "E", -1: "W", 0: "E"}
        latSign = {1: "N", -1: "S", 0: "N"}
        lonStr = f"{lonSign[np.sign(lon)]}{abs(lon):03}"
        latStr = f"{latSign[np.sign(lat)]}{abs(lat):02}"
        return f"s3://copernicus-dem-30m/Copernicus_DSM_COG_10_{latStr}_00_{lonStr}_00_DEM/Copernicus_DSM_COG_10_{latStr}_00_{lonStr}_00_DEM.tif"
    
    
    def process_shapefile(shapefile_path):
        shapefile_name = os.path.splitext(os.path.basename(shapefile_path))[0]
        download_path = os.path.join(download_base_path, shapefile_name)
        os.makedirs(download_path, exist_ok=True)
        aoi = gpd.read_file(shapefile_path).to_crs(destination_crs)
        for row in aoi.itertuples():
            xmin, ymin, xmax, ymax = row.geometry.bounds
            xmin -= 0.05
            ymin -= 0.05
            xmax += 0.05
            ymax += 0.05
            for lon in range(int(np.floor(xmin)), int(np.floor(xmax)) + 1):
                for lat in range(int(np.floor(ymin)), int(np.floor(ymax)) + 1):
                    myPath = s3Path(lat, lon)
                    myFile = os.path.basename(myPath)
                    cache_file = os.path.join(cache_folder, myFile)
                    local_file = os.path.join(download_path, myFile)
                    if os.path.exists(local_file):
                        print(f"[SKIP] Exists: {myFile}")
                        continue
                    if os.path.exists(cache_file):
                        print(f"[COPY] {myFile} from cache")
                        shutil.copy(cache_file, local_file)
                    else:
                        print(f"[DOWNLOAD] {myPath}")
                        cmd = f"aws s3 cp {myPath} {cache_file} --no-sign-request"
                        if call(cmd, shell=True) == 0:
                            shutil.copy(cache_file, local_file)
                        else:
                            print(f"[ERROR] Download failed: {myPath}")
        print(f"[DONE] DEMs ready for {shapefile_name}")
    
    Parallel(n_jobs=cfg.n_jobs)(
        delayed(process_shapefile)(shp)
        for shp in glob.glob(os.path.join(cfg.shapefile_dir, "*.shp"))
    )
    
    # === Step 3: Mosaic DEMs ===
    # === Step 3: Mosaic DEMs ===
    def mosaic_dems(folder):
        out_fp = os.path.join(folder, "mosaic_dem.tif")
        if os.path.exists(out_fp):
            print(f"[SKIP] Mosaic already exists: {out_fp}")
            return
    
        tif_files = glob.glob(os.path.join(folder, "*.tif"))
    
        if len(tif_files) == 0:
            print(f"[WARNING] No DEM files found in {folder}")
            return
    
        elif len(tif_files) == 1:
            print(f"[INFO] Only one DEM found in {folder}, copying as mosaic: {tif_files[0]}")
            shutil.copy(tif_files[0], out_fp)
            return
    
        try:
            mosaic, out_trans = merge([rasterio.open(f) for f in tif_files])
            meta = rasterio.open(tif_files[0]).meta.copy()
            meta.update({
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans
            })
    
            with rasterio.open(out_fp, "w", **meta) as dest:
                dest.write(mosaic)
    
            print(f"[SAVED] {out_fp}")
        except Exception as e:
            print(f"[ERROR] Failed to mosaic in {folder}: {e}")
    
    Parallel(n_jobs=cfg.n_jobs)(
        delayed(mosaic_dems)(os.path.join(download_base_path, d))
        for d in os.listdir(download_base_path)
    )
    
    
    
    # === Step 4: Cleanup DEMs ===
    def cleanup(folder):
        mosaic_fp = os.path.join(folder, "mosaic_dem.tif")
        if os.path.exists(mosaic_fp):
            for f in glob.glob(os.path.join(folder, "*.tif")):
                if f != mosaic_fp:
                    os.remove(f)
                    print(f"Deleted {f}")
    
    Parallel(n_jobs=cfg.n_jobs)(
        delayed(cleanup)(os.path.join(download_base_path, d))
        for d in os.listdir(download_base_path)
    )
    
    # === Step 5: Clip DEM to match images ===
    def reproject_and_resample(dem_file, ref_img, out_fp):
        with rasterio.open(ref_img) as ref:
            ref_crs, ref_transform = ref.crs, ref.transform
            width, height = ref.width, ref.height
        with rasterio.open(dem_file) as src:
            profile = src.meta.copy()
            profile.update({"crs": ref_crs, "transform": ref_transform, "width": width, "height": height})
            Path(os.path.dirname(out_fp)).mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_fp, "w", **profile) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=ref_transform,
                        dst_crs=ref_crs,
                        resampling=Resampling.bilinear
                    )
    
    def clip_dem_to_img(dem_fp, img_fp, out_fp):
        with rasterio.open(img_fp) as src:
            bbox = box(*src.bounds)
            shapes = [bbox]
        with rasterio.open(dem_fp) as dem:
            out_img, out_transform = mask(dem, shapes, crop=True)
            out_meta = dem.meta.copy()
            out_meta.update({"height": out_img.shape[1], "width": out_img.shape[2], "transform": out_transform})
            with rasterio.open(out_fp, "w", **out_meta) as dst:
                dst.write(out_img)
    
    for tif in glob.glob(os.path.join(cfg.s2_images, "*.tif")):
        img_name = os.path.splitext(os.path.basename(tif))[0]
        dem_folder = os.path.join(download_base_path, img_name)
        reprojected = os.path.join(dem_folder, "dem_proj.tif")
        clipped = os.path.join(dem_folder, "DEM.tif")
        if not os.path.exists(os.path.join(dem_folder, "mosaic_dem.tif")):
            print(f"[SKIP] No mosaic found for {img_name}")
            continue
        reproject_and_resample(os.path.join(dem_folder, "mosaic_dem.tif"), tif, reprojected)
        clip_dem_to_img(reprojected, tif, clipped)
    
    # === Step 6: Compute slope from clipped DEM ===
    def compute_slope(dem_fp, out_fp):
        with rasterio.open(dem_fp) as src:
            dem = src.read(1).astype(float)
            transform = src.transform
            nodata = src.nodata
        xres, yres = transform.a, -transform.e
        dzdx, dzdy = np.gradient(dem, xres, yres)
        slope = np.sqrt(dzdx**2 + dzdy**2)
        slope_deg = np.arctan(slope) * (180 / np.pi)
        if nodata is not None:
            slope_deg[dem == nodata] = nodata
        profile = src.meta.copy()
        profile.update({"dtype": "float32"})
        with rasterio.open(out_fp, "w", **profile) as dst:
            dst.write(slope_deg.astype(np.float32), 1)
    
    for tif in glob.glob(os.path.join(download_base_path, "*", "DEM.tif")):
        slope_fp = os.path.join(os.path.dirname(tif), "slope.tif")
        compute_slope(tif, slope_fp)
    
    # === Step 7: Stack S2 + DEM + slope ===
    def stack_layers(image_fp, slope_fp, dem_fp, out_fp):
        with rasterio.open(image_fp) as src:
            img = src.read()
            meta = src.meta.copy()
        with rasterio.open(slope_fp) as s:
            slope = s.read(1)
        with rasterio.open(dem_fp) as d:
            dem = d.read(1)
        stacked = np.vstack([img, slope[np.newaxis, ...], dem[np.newaxis, ...]])
        meta.update({"count": stacked.shape[0], "dtype": "float32"})
        Path(os.path.dirname(out_fp)).mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_fp, "w", **meta) as dst:
            dst.write(stacked.astype(np.float32))
    
    for tif in glob.glob(os.path.join(cfg.s2_images, "*.tif")):
        img_name = os.path.basename(tif)
        dem_folder = os.path.join(download_base_path, os.path.splitext(img_name)[0])
        dem_fp = os.path.join(dem_folder, "DEM.tif")
        slope_fp = os.path.join(dem_folder, "slope.tif")
        out_fp = os.path.join(cfg.stacked_dir, img_name)
        if os.path.exists(dem_fp) and os.path.exists(slope_fp):
            stack_layers(tif, slope_fp, dem_fp, out_fp)
            
    # === Final Check: Identify Unprocessed Images ===
    input_images = sorted([os.path.splitext(f)[0] for f in os.listdir(cfg.s2_images) if f.endswith(".tif")])
    output_images = sorted([os.path.splitext(f)[0] for f in os.listdir(cfg.stacked_dir) if f.endswith(".tif")])
    
    missing_images = sorted(set(input_images) - set(output_images))
    
    if missing_images:
        print("\n[WARNING] The following images were not processed and are missing in the stacked output:")
        for fname in missing_images:
            print(f" - {fname}.tif")
    else:
        print("\n[OK] All Sentinel-2 images successfully stacked.")
    
    
    print("\n[COMPLETE] Script finished in %.2f seconds" % (time.time() - start_time))
    