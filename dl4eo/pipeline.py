from dl4eo.config import PipelineConfig
from dl4eo.stages import (
    download_sentinel2,
    preprocess_s2,
    generate_aoi,
    prepare_dem,
    prepare_sentinel1,
    generate_mask,
    normalize_data,
)

def generate_dataset(base_dir, aoi_shapefile_dir, feature_shapefile, date_range,
                     cloud_cover=20, box_size_m=2560, n_jobs=8):
    cfg = PipelineConfig(
        base_dir=base_dir,
        aoi_shapefile_dir=aoi_shapefile_dir,
        feature_shapefile=feature_shapefile,
        date_range=date_range,
        cloud_cover=cloud_cover,
        box_size_m=box_size_m,
        n_jobs=n_jobs
    )

    print("\n[STAGE] Downloading Sentinel-2")
    download_sentinel2.run(cfg)

    print("\n[STAGE] Preprocessing S2")
    preprocess_s2.run(cfg)

    print("\n[STAGE] Generating AOIs")
    generate_aoi.run(cfg)

    print("\n[STAGE] Preparing DEM")
    prepare_dem.run(cfg)

    print("\n[STAGE] Processing Sentinel-1")
    prepare_sentinel1.run(cfg)

    print("\n[STAGE] Generating Masks")
    generate_mask.run(cfg)

    print("\n[STAGE] Normalizing Data")
    normalize_data.run(cfg)

    print("\n✅ Pipeline complete.")