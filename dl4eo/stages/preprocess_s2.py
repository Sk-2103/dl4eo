def run(cfg):
    # === START OF SCRIPT ===
    import time
    import os
    import shutil
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from joblib import Parallel, delayed
    import glob
    import numpy as np

    print("=" * 60)
    print(f"[START] {__file__} running...")
    start_time = time.time()

    # Directories from cfg
    input_dir = cfg.s2_images
    output_dir = cfg.s2_raw

    def process_folder(root, files):
        green_files = [file for file in files if file.endswith('_green.tif')]
        if not green_files:
            return

        ref_image_path = os.path.join(root, green_files[0])
        output_subdir = os.path.join(output_dir, os.path.relpath(root, input_dir))

        # Skip folder if already processed
        expected_bands = ['blue.tif', 'green.tif', 'red.tif', 'nir.tif', 'swir16.tif', 'swir22.tif']
        if os.path.exists(output_subdir):
            existing_files = os.listdir(output_subdir)
            if all(any(f.endswith(b) for f in existing_files) for b in expected_bands):
                print(f"[SKIP] Folder already processed: {output_subdir}")
                return

        os.makedirs(output_subdir, exist_ok=True)

        def resample_tif(file_path, output_file, ref_image_path):
            try:
                with rasterio.open(file_path) as src, rasterio.open(ref_image_path) as ref_src:
                    ref_transform = ref_src.transform
                    ref_height, ref_width = ref_src.height, ref_src.width
                    crs = src.crs
                    data = src.read()

                    with rasterio.open(output_file, 'w', driver='GTiff',
                                       width=ref_width, height=ref_height, count=data.shape[0], dtype=data.dtype,
                                       crs=crs, transform=ref_transform) as dst:
                        for i in range(data.shape[0]):
                            reproject(
                                source=data[i],
                                destination=rasterio.band(dst, i + 1),
                                src_transform=src.transform,
                                src_crs=crs,
                                dst_transform=ref_transform,
                                dst_crs=crs,
                                resampling=Resampling.bilinear
                            )
                print(f"Resampled: {file_path} -> {output_file}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

        tasks = []
        for file in files:
            file_path = os.path.join(root, file)
            output_file = os.path.join(output_subdir, file)

            if os.path.abspath(file_path) == os.path.abspath(output_file):
                print(f"[SKIP] {file_path} already in destination")
                continue

            if file.endswith(('blue.tif', 'green.tif', 'red.tif', 'nir.tif')):
                shutil.copy2(file_path, output_file)
                print(f"Copied: {file_path} -> {output_file}")
            elif file.endswith(('swir16.tif', 'swir22.tif', 'rededge1.tif')):
                tasks.append(delayed(resample_tif)(file_path, output_file, ref_image_path))

        if tasks:
            Parallel(n_jobs=cfg.n_jobs, prefer="threads")(tasks)

    all_folders = [(root, files) for root, _, files in os.walk(input_dir)]
    Parallel(n_jobs=cfg.n_jobs, prefer="threads")(delayed(process_folder)(root, files) for root, files in all_folders)
    print("Processing complete!")

    # === NDWI Calculation ===
    def calculate_ndwi(green, nir):
        green = green.astype('float32') / 65535.0
        nir = nir.astype('float32') / 65535.0
        return (green - nir) / (green + nir + 1e-6)

    def calculate_and_save_ndwi(folder_path):
        print(f"📂 Processing folder: {folder_path}")
        ndwi_path = os.path.join(folder_path, "NDWI.tif")
        if os.path.exists(ndwi_path):
            print(f"[SKIP] NDWI already exists at: {ndwi_path}")
            return

        green_path = glob.glob(os.path.join(folder_path, "*_green.tif"))
        nir_path = glob.glob(os.path.join(folder_path, "*_nir.tif"))

        if not green_path or not nir_path:
            print(f"⚠ Missing Green or NIR band in {folder_path}")
            return

        with rasterio.open(green_path[0]) as green_ds, rasterio.open(nir_path[0]) as nir_ds:
            green = green_ds.read(1)
            nir = nir_ds.read(1)
            profile = green_ds.profile

        ndwi = calculate_ndwi(green, nir)
        profile.update(dtype=rasterio.uint16, count=1, compress='lzw')

        with rasterio.open(ndwi_path, 'w', **profile) as dst:
            dst.write((ndwi * 65535).astype(np.uint16), 1)
        print(f"✅ NDWI saved at: {ndwi_path}")

    ndwi_folders = [os.path.join(dp, d) for dp, dn, _ in os.walk(cfg.s2_raw) for d in dn if d.startswith("S2A") or d.startswith("S2B")]
    Parallel(n_jobs=cfg.n_jobs, prefer="threads")(delayed(calculate_and_save_ndwi)(folder) for folder in ndwi_folders)
    print("🎉 All NDWI processing complete!")

    # === Band Stacking ===
    band_suffixes = ['blue', 'green', 'red', 'nir', 'swir16', 'swir22', 'NDWI']
    stack_output = cfg.s2_stack
    os.makedirs(stack_output, exist_ok=True)

    def process_stack(satellite_folder):
        folder_name = os.path.basename(satellite_folder)
        out_path = os.path.join(stack_output, f"{folder_name}.tif")
        if os.path.exists(out_path):
            print(f"[SKIP] Stack already exists: {out_path}")
            return

        print(f"🔄 Stacking: {folder_name}")
        band_map = {b: None for b in band_suffixes}
        for f in os.listdir(satellite_folder):
            f_lower = f.lower()
            for band in band_suffixes:
                if band.lower() in f_lower and f_lower.endswith('.tif'):
                    band_map[band] = os.path.join(satellite_folder, f)

        selected_paths = [band_map[b] for b in band_suffixes if band_map[b] is not None]
        if len(selected_paths) < len(band_suffixes):
            print(f"⚠ Skipping {folder_name} (missing required bands)")
            return

        try:
            with rasterio.open(selected_paths[0]) as ref:
                meta = ref.meta.copy()
                height, width = ref.height, ref.width
                dtype = ref.read(1).dtype

            stack = np.empty((len(selected_paths), height, width), dtype=dtype)
            for i, path in enumerate(selected_paths):
                with rasterio.open(path) as src:
                    stack[i] = src.read(1)

            meta.update(count=len(selected_paths))
            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write(stack)

            print(f"✅ Saved: {out_path}")
        except Exception as e:
            print(f"❌ Error in {folder_name}: {e}")

    Parallel(n_jobs=cfg.n_jobs, prefer="threads")(delayed(process_stack)(folder) for folder in ndwi_folders)
    print("🎉 All stacking complete!")

    # === Cleanup Raw Input ===
    print("\n🧹 Cleaning up original raw folders under:", input_dir)

    try:
        for root, dirs, _ in os.walk(input_dir, topdown=False):
            for dir_name in dirs:
                full_path = os.path.join(root, dir_name)
                if dir_name.startswith("S2A") or dir_name.startswith("S2B"):
                    try:
                        shutil.rmtree(full_path)
                        print(f"[DELETED] {full_path}")
                    except Exception as e:
                        print(f"[ERROR] Could not delete {full_path}: {e}")
    except Exception as cleanup_error:
        print(f"[FATAL] Error during cleanup: {cleanup_error}")

    end_time = time.time()
    print(f"[DONE] {__file__} completed in {end_time - start_time:.2f} seconds")
    print("=" * 60)
