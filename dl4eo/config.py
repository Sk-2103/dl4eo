import os
from dataclasses import dataclass, field
from typing import List, Optional

# Spectral index definitions: (band1, band2) used in (band1 - band2) / (band1 + band2)
SPECTRAL_INDICES = {
    "NDWI":  ("green", "nir"),       # Normalized Difference Water Index
    "NDSI":  ("green", "swir16"),    # Normalized Difference Snow Index
    "NDVI":  ("nir", "red"),         # Normalized Difference Vegetation Index
    "NDRE":  ("nir", "rededge1"),    # Normalized Difference Red Edge
    "MNDWI": ("green", "swir16"),    # Modified NDWI (same as NDSI for S2)
    "EVI":   None,                   # Enhanced Vegetation Index (special formula)
}

DEFAULT_S2_BANDS = ["blue", "green", "red", "nir", "swir16", "swir22"]


@dataclass
class PipelineConfig:
    # --- Required ---
    base_dir: str
    aoi_shapefile_dir: str
    feature_shapefile: str
    date_range: str

    # --- Sentinel-2 ---
    cloud_cover: int = 20
    s2_bands: List[str] = field(default_factory=lambda: list(DEFAULT_S2_BANDS))
    spectral_index: Optional[str] = "NDWI"   # None to skip index computation

    # --- Patch generation ---
    patch_size: int = 256          # output patch size in pixels
    resolution_m: int = 10        # native S2 resolution used as reference
    overlap: float = 0.0          # fractional overlap between patches [0, 1)

    # --- Sentinel-1 temporal matching ---
    # Search window: target_s2_date ± sar_days_delta days.
    # Increase if SAR coverage is sparse in your region (e.g. 10–15 for high latitudes).
    # The closest available SAR acquisition within the window is selected.
    sar_days_delta: int = 5

    # --- Execution ---
    n_jobs: int = 8
    skip_sentinel1: bool = False
    skip_dem: bool = False

    # --- Normalization ---
    # Kept for users who need pre-normalised files (e.g. non-PyTorch workflows).
    # Recommended: leave False and use dl4eo.stats + dl4eo.io.PatchDataset instead.
    normalize: bool = False
    norm_mode: str = "per_band"  # "per_band" | "per_modality" | "none"

    # --- Mask generation ---
    multi_class: bool = False
    class_field: Optional[str] = None   # shapefile attribute holding class IDs

    def __post_init__(self):
        if self.spectral_index and self.spectral_index not in SPECTRAL_INDICES:
            raise ValueError(
                f"Unknown spectral_index '{self.spectral_index}'. "
                f"Supported: {list(SPECTRAL_INDICES)}"
            )
        if not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be in [0, 1)")
        if self.sar_days_delta < 1:
            raise ValueError("sar_days_delta must be ≥ 1")

        # Derive box size from patch_size × resolution
        self.box_size_m = self.patch_size * self.resolution_m

        # Stride between patches (in metres)
        self.stride_m = int(self.box_size_m * (1.0 - self.overlap))

        # Aliases kept for backward compat
        self.shapefile_dir = self.aoi_shapefile_dir
        self.lake_shp_path = self.feature_shapefile

        # Directory layout under base_dir
        self.s2_images         = os.path.join(self.base_dir, "images")
        self.s2_raw            = os.path.join(self.base_dir, "Resampled")
        self.s2_stack          = os.path.join(self.base_dir, "stack")
        self.aoi_boxes         = os.path.join(self.base_dir, "AOI_boxes")
        self.dem_dir           = os.path.join(self.base_dir, "DEM")
        self.sar_dir           = os.path.join(self.base_dir, "GRD")
        self.sar_clipped       = os.path.join(self.base_dir, "Clipped_SAR")
        self.stacked_dir       = os.path.join(self.base_dir, "stacked")
        self.stacked_sar       = os.path.join(self.base_dir, "stacked_with_sar")
        self.stacked_with_sar_dir = self.stacked_sar
        self.normalized        = os.path.join(self.base_dir, "normalized")
        self.masks             = os.path.join(self.base_dir, "mask")
        self.shapefile_each    = os.path.join(self.base_dir, "shapefile", "each")
        self.stacked_sample_wgs84 = os.path.join(self.base_dir, "stacked_sample_wgs84")

        # Bands needed for the selected spectral index
        self._index_bands: List[str] = []
        if self.spectral_index:
            spec = SPECTRAL_INDICES[self.spectral_index]
            if spec is not None:
                self._index_bands = list(spec)
            else:
                # EVI needs blue, red, nir
                self._index_bands = ["blue", "red", "nir"]

        # Full set of bands to download (s2_bands + any extras for index)
        self.bands_to_download = list(
            dict.fromkeys(self.s2_bands + self._index_bands)
        )

    def describe(self) -> str:
        """Human-readable summary of pipeline configuration."""
        lines = [
            "DL4EO Pipeline Configuration",
            "=" * 40,
            f"  AOI shapefile dir : {self.aoi_shapefile_dir}",
            f"  Feature shapefile : {self.feature_shapefile}",
            f"  Date range        : {self.date_range}",
            f"  Cloud cover < {self.cloud_cover}%",
            f"  S2 bands          : {self.s2_bands}",
            f"  Spectral index    : {self.spectral_index}",
            f"  Patch size        : {self.patch_size}px × {self.patch_size}px "
            f"({self.box_size_m}m × {self.box_size_m}m)",
            f"  Overlap           : {self.overlap * 100:.0f}%",
            f"  Skip Sentinel-1   : {self.skip_sentinel1}",
            f"  SAR days delta    : ±{self.sar_days_delta} days (S2 acq ± window)",
            f"  Skip DEM          : {self.skip_dem}",
            f"  Normalize         : {self.normalize} ({self.norm_mode})",
            f"  Multi-class mask  : {self.multi_class}",
            f"  Parallel jobs     : {self.n_jobs}",
            f"  Output directory  : {self.base_dir}",
        ]
        return "\n".join(lines)
