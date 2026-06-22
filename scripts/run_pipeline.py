"""
Command-line entry point for the DL4EO pipeline.

Example:
    python scripts/run_pipeline.py \
        --base_dir /output/glacial_lakes \
        --aoi /data/AOI/ \
        --features /data/lake_boundaries.shp \
        --date 2021-06-01/2021-08-31 \
        --cloud 20 \
        --patch_size 256 \
        --index NDWI \
        --overlap 0.25 \
        --jobs 8
"""
import argparse
import dl4eo

parser = argparse.ArgumentParser(description="DL4EO Dataset Builder")
parser.add_argument("--base_dir",    required=True,  help="Output root directory")
parser.add_argument("--aoi",         required=True,  help="AOI shapefile directory")
parser.add_argument("--features",    required=True,  help="Label shapefile path")
parser.add_argument("--date",        required=True,  help="Date range YYYY-MM-DD/YYYY-MM-DD")
parser.add_argument("--cloud",       type=int,   default=20,    help="Max cloud cover %%")
parser.add_argument("--patch_size",  type=int,   default=256,   help="Patch size in pixels")
parser.add_argument("--resolution",  type=int,   default=10,    help="Reference resolution in metres")
parser.add_argument("--overlap",     type=float, default=0.0,   help="Patch overlap fraction [0,1)")
parser.add_argument("--index",       default="NDWI",            help="Spectral index (NDWI/NDSI/NDVI/NDRE/EVI/none)")
parser.add_argument("--bands",       nargs="+",  default=None,  help="S2 band names to include")
parser.add_argument("--jobs",        type=int,   default=8,     help="Parallel workers")
parser.add_argument("--sar_delta",   type=int,   default=5,     help="Sentinel-1 search window: S2 date ± N days (default 5)")
parser.add_argument("--no-s1",       action="store_true",       help="Skip Sentinel-1 SAR")
parser.add_argument("--no-dem",      action="store_true",       help="Skip DEM")
parser.add_argument("--norm_mode",   default="per_band",        help="per_band|per_modality|none (use PatchDataset norm instead)")
parser.add_argument("--multi_class", action="store_true",       help="Multi-class mask mode")
parser.add_argument("--class_field", default=None,              help="Shapefile field for class IDs")
args = parser.parse_args()

spectral_index = None if args.index.lower() == "none" else args.index

dl4eo.generate_dataset(
    base_dir=args.base_dir,
    aoi_shapefile_dir=args.aoi,
    feature_shapefile=args.features,
    date_range=args.date,
    cloud_cover=args.cloud,
    patch_size=args.patch_size,
    resolution_m=args.resolution,
    overlap=args.overlap,
    spectral_index=spectral_index,
    s2_bands=args.bands,
    sar_days_delta=args.sar_delta,
    skip_sentinel1=args.no_s1,
    skip_dem=args.no_dem,
    multi_class=args.multi_class,
    class_field=args.class_field,
    n_jobs=args.jobs,
)
