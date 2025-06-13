def run(cfg):
    import time
    
    print("="*60)
    print(f"[START] {__file__} running...")
    start_time = time.time()
    
    
    import os
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from shapely.geometry import Polygon, MultiPolygon, shape as shapely_shape
    from shapely.ops import unary_union
    from matplotlib.lines import Line2D
    from pystac_client import Client
    
    collection = "sentinel-2-l2a"
    date_range = "2020-08-01/2020-08-30"
    filters = {"eo:cloud_cover": {"lt": 20}}
    
    client = Client.open("https://earth-search.aws.element84.com/v1")
    
    def remove_duplicate_coordinates(geom):
        if isinstance(geom, (Polygon, MultiPolygon)):
            coords = list(geom.exterior.coords)
            cleaned = [coords[0]] + [c for i, c in enumerate(coords[1:]) if c != coords[i]]
            return Polygon(cleaned)
        return geom
    
    def compute_aoi_coverage(aoi, items):
        total_aoi_area_km2 = aoi.area * (111 ** 2)
        intersections = []
    
        for item in items:
            img_geom = shapely_shape(item.geometry)
            inter = aoi.intersection(img_geom)
            if not inter.is_empty:
                intersections.append(inter)
    
        if not intersections:
            print("[COVERAGE] No intersecting image footprints.")
            return 0, 0, 0, []
    
        union_geom = unary_union(intersections)
        covered_area_km2 = union_geom.area * (111 ** 2)
        coverage_pct = 100 * covered_area_km2 / total_aoi_area_km2
    
        print(f"[COVERAGE] AOI Area: {total_aoi_area_km2:.2f} km²")
        print(f"[COVERAGE] Covered Area: {covered_area_km2:.2f} km²")
        print(f"[COVERAGE] Percentage Coverage: {coverage_pct:.2f}%")
        return total_aoi_area_km2, covered_area_km2, coverage_pct, intersections
    
    def plot_coverage_map(aoi, items, intersections, name):
        img_geoms = []
        cloud_covers = []
    
        for item in items:
            img_geoms.append(shapely_shape(item.geometry))
            cloud_covers.append(item.properties.get("eo:cloud_cover", 0))
    
        if not img_geoms:
            print(f"[WARNING] No valid image geometries to plot for {name}")
            return
    
        gdf_imgs = gpd.GeoDataFrame({'geometry': img_geoms, 'cloud': cloud_covers}, crs="EPSG:4326")
        gdf_aoi = gpd.GeoDataFrame(geometry=[aoi], crs="EPSG:4326")
        gdf_inter = gpd.GeoDataFrame(geometry=intersections, crs="EPSG:4326")
        gdf_lakes = gpd.read_file(cfg.lake_shp_path).to_crs("EPSG:4326")
    
        norm = Normalize(vmin=0, vmax=max(cloud_covers) if cloud_covers else 20)
        cmap = colormaps["Greens"]
    
        fig, ax = plt.subplots(figsize=(10, 8))
    
        # Plot layers
        gdf_aoi.boundary.plot(ax=ax, edgecolor='black', linestyle='--', linewidth=1.5)
        gdf_inter.plot(ax=ax, color='lightgreen', alpha=0.4)
        gdf_imgs.plot(ax=ax, edgecolor='black', linewidth=0.5,
                      facecolor=gdf_imgs['cloud'].apply(lambda c: cmap(norm(c))))
        gdf_lakes.plot(ax=ax, edgecolor='blue', linewidth=1)
    
        # Colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm._A = []
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5)
        cbar.set_label('Cloud Cover (%)')
    
        # Manual legend
        legend_elements = [
            Line2D([0], [0], color='black', linestyle='--', lw=1.5, label='AOI'),
            Line2D([0], [0], color='lightgreen', lw=8, label='Intersected Area'),
            Line2D([0], [0], color='green', lw=4, label='Image Footprints'),
            Line2D([0], [0], color='blue', lw=2, label='Lakes'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
        ax.set_title(f"AOI Coverage Map: {name}", fontsize=14)
        ax.set_axis_off()
        plt.tight_layout()
    
        try:
            plt.show()
        except:
            plt.savefig(f"{name}_coverage_map.png", dpi=300)
            print(f"[SAVED] Plot saved to {name}_coverage_map.png")
    
    def visualize_shapefiles(shapefile_dir):
        shapefiles = [os.path.join(cfg.shapefile_dir, f) for f in os.listdir(cfg.shapefile_dir) if f.endswith(".shp")]
        for shapefile_path in shapefiles:
            name = os.path.splitext(os.path.basename(shapefile_path))[0]
            print(f"\n[START] Visualizing shapefile: {name}")
    
            gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
            valid_geoms = [remove_duplicate_coordinates(g) for g in gdf.geometry if g.is_valid]
    
            if not valid_geoms:
                print(f"[SKIP] No valid or non-empty geometry in {name}")
                continue
    
            # Use union of all geometries as AOI
            aoi = unary_union(valid_geoms)
            aoi_geojson = aoi.__geo_interface__
    
            search = client.search(
                collections=[collection],
                intersects=aoi_geojson,
                datetime=date_range,
                query=filters
            )
    
            items = list(search.item_collection())
            if not items:
                print(f"[INFO] No scenes found for {name}")
                continue
    
            print(f"[INFO] Matching scenes: {len(items)}")
            _, _, _, intersections = compute_aoi_coverage(aoi, items)
            plot_coverage_map(aoi, items, intersections, name)
    
    # === Run ===
    visualize_shapefiles(cfg.shapefile_dir)
    
    import requests
    import shutil
    
    def process_scene(item, local_dir):
        try:
            scene_dir = os.path.join(local_dir, item.id)
            os.makedirs(scene_dir, exist_ok=True)
    
            band_alias = {
                'B01': 'coastal', 'B02': 'blue', 'B03': 'green',
                'B04': 'red', 'B05': 'rededge1', 'B06': 'rededge2',
                'B07': 'rededge3', 'B08': 'nir', 'B8A': 'narrow_nir',
                'B09': 'water_vapor', 'B11': 'swir1', 'B12': 'swir2',
                'visual': 'rgb'
            }
    
            for asset_key, asset in item.assets.items():
                if asset_key in ['thumbnail', 'tileinfo_metadata', 'granule_metadata']:
                    continue
    
                url = asset.href
                suffix = band_alias.get(asset_key, asset_key.lower())
                filename = f"{item.id}_{suffix}.tif"
                local_path = os.path.join(scene_dir, filename)
    
                if os.path.exists(local_path):
                    print(f"[SKIP] Already exists: {local_path}")
                    continue
    
                print(f"[DOWNLOAD] {url} → {local_path}")
                try:
                    with requests.get(url, stream=True, timeout=120) as r:
                        if r.status_code != 200:
                            print(f"[ERROR] {r.status_code} - Failed to download {url}")
                            continue
                        with open(local_path, 'wb') as f:
                            shutil.copyfileobj(r.raw, f)
    
                    if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
                        print(f"[DONE] {filename} ✓")
                    else:
                        print(f"[ERROR] Empty or incomplete file: {filename}")
                except Exception as e:
                    print(f"[ERROR] Exception while downloading {url}: {e}")
        except Exception as e:
            print(f"[ERROR] Scene {item.id}: {e}")
    
    import concurrent.futures
    
    def download_data(shapefile_dir, local_dir_base):
        shapefiles = [f for f in os.listdir(cfg.shapefile_dir) if f.endswith(".shp")]
    
        def process_shapefile(shapefile_path):
            name = os.path.splitext(os.path.basename(shapefile_path))[0]
            print(f"\n[DOWNLOAD] Processing shapefile: {name}")
            gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")
            valid_geoms = [remove_duplicate_coordinates(g) for g in gdf.geometry if g.is_valid]
            if not valid_geoms:
                print(f"[SKIP] No valid geometry in {name}")
                return
            aoi = valid_geoms[0]
            aoi_geojson = aoi.__geo_interface__
    
            search = client.search(
                collections=[collection],
                intersects=aoi_geojson,
                datetime=date_range,
                query=filters
            )
            items = list(search.get_all_items())
            if not items:
                print(f"[INFO] No scenes to download for {name}")
                return
    
            local_dir = os.path.join(local_dir_base, name)
            os.makedirs(local_dir, exist_ok=True)
            for item in items:
                process_scene(item, local_dir)
    
        full_paths = [os.path.join(cfg.shapefile_dir, f) for f in shapefiles]
    
        with concurrent.futures.ThreadPoolExecutor(max_workers=80) as executor:
            executor.map(process_shapefile, full_paths)
    
    
    # Define the output directory for downloaded scenes
    #out_dir = os.path.join(cfg.s2_images, scene_id)
    
    # Call the downloader after you've visually verified the coverage
    download_data(cfg.shapefile_dir, cfg.s2_images)
    
    print("download complete")
    
    end_time = time.time()
    print(f"[DONE] {__file__} completed in {end_time - start_time:.2f} seconds")
    print("="*60)