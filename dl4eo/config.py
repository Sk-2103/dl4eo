import os
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    base_dir: str
    aoi_shapefile_dir: str  # ✅ this is what your code is calling
    feature_shapefile: str
    date_range: str
    cloud_cover: int = 20
    box_size_m: int = 2560
    n_jobs: int = 8

    def __post_init__(self):
        self.shapefile_dir = self.aoi_shapefile_dir
        self.lake_shp_path = self.feature_shapefile

        self.s2_images = os.path.join(self.base_dir, "images")
        self.s2_raw = os.path.join(self.base_dir, "Resampled")
        self.s2_stack = os.path.join(self.base_dir, "stack")
        self.aoi_boxes = os.path.join(self.base_dir, "AOI_boxes")
        self.dem_dir = os.path.join(self.base_dir, "DEM")
        self.sar_dir = os.path.join(self.base_dir, "GRD")
        self.sar_extracted_dir = os.path.join(self.base_dir, "GRD_Extracted")
        self.sar_clipped = os.path.join(self.base_dir, "Clipped_SAR")
        self.stacked_sar = os.path.join(self.base_dir, "stacked_with_sar")
        self.normalized = os.path.join(self.base_dir, "normalize")
        self.masks = os.path.join(self.base_dir, "mask")
        self.stacked_dir = os.path.join(self.base_dir, "stacked")
        self.shapefile_each = os.path.join(self.base_dir, "shapefile", "each")
        self.stacked_sample_wgs84 = os.path.join(self.base_dir, "stacked_sample_wgs84")
        self.stacked_with_sar_dir = self.stacked_sar
