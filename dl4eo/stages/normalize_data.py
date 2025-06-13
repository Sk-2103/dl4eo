def run(cfg):
    import os
    import time
    import numpy as np
    import rasterio
    from joblib import Parallel, delayed
    
    print("=" * 60)
    print(f"[START] {__file__} running...")
    start_time = time.time()
    
    def clean_minmax_normalize(input_path, output_path):
        """Per-band min-max normalization. Removes nodata, normalizes to [0, 1], fills masked with 0."""
        try:
            with rasterio.open(input_path) as src:
                meta = src.meta.copy()
                meta.update(dtype=rasterio.float32)
                meta.pop('nodata', None)  # remove nodata from output
    
                normalized_bands = []
    
                for i in range(1, src.count + 1):
                    data = src.read(i)
                    nodata_val = src.nodata
                    if nodata_val is not None:
                        mask = (data == nodata_val) | np.isnan(data)
                    else:
                        mask = np.isnan(data)
    
                    valid = data[~mask]
    
                    norm_band = np.zeros_like(data, dtype=np.float32)  # fill masked with 0
    
                    if valid.size > 0:
                        min_val = valid.min()
                        max_val = valid.max()
                        if max_val > min_val:
                            norm_band[~mask] = (data[~mask] - min_val) / (max_val - min_val)
                        else:
                            norm_band[~mask] = 0.0  # flat image
    
                    normalized_bands.append(norm_band)
    
                with rasterio.open(output_path, 'w', **meta) as dst:
                    for i, band in enumerate(normalized_bands, start=1):
                        dst.write(band, i)
    
            print(f"✅ Min-Max Normalized: {os.path.basename(input_path)}")
    
        except Exception as e:
            print(f"❌ Error processing {input_path}: {e}")
    
    def normalize_folder(input_folder, output_folder, n_jobs=6):
        os.makedirs(output_folder, exist_ok=True)
        file_list = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
    
        print(f"📁 Found {len(file_list)} raster files in: {input_folder}")
        print(f"🚀 Starting clean min-max normalization with {cfg.n_jobs} workers...\n")
    
        Parallel(n_jobs=cfg.n_jobs)(
            delayed(clean_minmax_normalize)(
                os.path.join(input_folder, fname),
                os.path.join(output_folder, fname)
            ) for fname in file_list
        )
    
        print("\n✅ All files processed and saved to:", output_folder)
    
    # === Configuration ===
    input_folder = cfg.stacked_sar
    output_folder = cfg.normalized  # final cfg.normalized output
    normalize_folder(input_folder, output_folder, n_jobs=cfg.n_jobs)
    
    end_time = time.time()
    print(f"[DONE] {__file__} completed in {end_time - start_time:.2f} seconds")
    print("=" * 60)