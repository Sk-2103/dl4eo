import argparse
from rsdl_pipeline.config import PipelineConfig
from rsdl_pipeline.pipeline import run_pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', required=True)
parser.add_argument('--aoi', required=True)
parser.add_argument('--features', required=True)
parser.add_argument('--date', required=True)
parser.add_argument('--cloud', type=int, default=20)
parser.add_argument('--box', type=int, default=2560)
parser.add_argument('--jobs', type=int, default=8)
args = parser.parse_args()

config = PipelineConfig(
    base_dir=args.base_dir,
    aoi_shapefile=args.aoi,
    feature_shapefile=args.features,
    date_range=args.date,
    cloud_cover=args.cloud,
    box_size_m=args.box,
    n_jobs=args.jobs
)

run_pipeline(config)