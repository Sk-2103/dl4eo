def run(cfg):
    import os
    import time
    import shutil
    import concurrent.futures
    import requests
    import geopandas as gpd
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    from pystac_client import Client

    print("=" * 60)
    print(f"[START] Downloading Sentinel-2 (L2A)")
    start_time = time.time()

    client = Client.open("https://earth-search.aws.element84.com/v1")
    collection = "sentinel-2-l2a"
    os.makedirs(cfg.s2_images, exist_ok=True)

    def _clean_geom(geom):
        if isinstance(geom, (Polygon, MultiPolygon)):
            if isinstance(geom, Polygon):
                coords = list(geom.exterior.coords)
                cleaned = [coords[0]] + [c for i, c in enumerate(coords[1:]) if c != coords[i]]
                return Polygon(cleaned)
        return geom

    # --- Download a single asset file ---
    def _download_asset(url, local_path):
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return
        try:
            with requests.get(url, stream=True, timeout=(10, 300)) as r:
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
        except Exception as e:
            print(f"[ERROR] Download failed {url}: {e}")

    # --- Bands needed: user-configured + index bands ---
    bands_needed = set(cfg.bands_to_download)

    # --- Process one AOI shapefile ---
    def process_shapefile(shp_path):
        name = os.path.splitext(os.path.basename(shp_path))[0]
        gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
        valid = [_clean_geom(g) for g in gdf.geometry if g and g.is_valid]
        if not valid:
            print(f"[SKIP] No valid geometry in {name}")
            return
        aoi = unary_union(valid)

        search = client.search(
            collections=[collection],
            intersects=aoi.__geo_interface__,
            datetime=cfg.date_range,
            query={"eo:cloud_cover": {"lt": cfg.cloud_cover}},
        )
        try:
            items = list(search.item_collection())
        except Exception:
            items = list(search.items())

        if not items:
            print(f"[INFO] No scenes found for {name}")
            return
        print(f"[INFO] {name}: {len(items)} scene(s) found")

        for item in items:
            scene_dir = os.path.join(cfg.s2_images, item.id)
            os.makedirs(scene_dir, exist_ok=True)

            for asset_key, asset in item.assets.items():
                if asset_key in {"thumbnail", "tileinfo_metadata", "granule_metadata",
                                  "visual", "overview", "SCL", "AOT", "WVP"}:
                    continue
                # Map common asset key variants to our band name scheme
                mapped = asset_key.lower().replace("-", "_")
                if mapped not in bands_needed:
                    continue

                ext = ".tif"
                filename = f"{item.id}_{mapped}{ext}"
                local_path = os.path.join(scene_dir, filename)
                if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                    print(f"[SKIP] {filename}")
                    continue
                print(f"[DOWNLOAD] {filename}")
                _download_asset(asset.href, local_path)

    # Exclude feature_shapefile so label files don't get used as AOI search geometries
    feature_shp_name = os.path.basename(cfg.feature_shapefile)
    shapefiles = [
        os.path.join(cfg.aoi_shapefile_dir, f)
        for f in os.listdir(cfg.aoi_shapefile_dir)
        if f.endswith(".shp") and f != feature_shp_name
    ]
    if not shapefiles:
        print(f"[WARN] No shapefiles found in {cfg.aoi_shapefile_dir}")
        return

    max_workers = min(cfg.n_jobs * 2, 16)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        ex.map(process_shapefile, shapefiles)

    end_time = time.time()
    print(f"[DONE] Sentinel-2 download complete in {end_time - start_time:.1f}s")
    print("=" * 60)
