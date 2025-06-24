# dl4eo

**dl4eo** is a Python package designed to streamline the end-to-end pipeline for generating multi-source remote sensing dataset for deep learning applications. This pipeline primarily combines  Sentinel-1, Sentinel-2 and Copernicus DEM. It automates downloading, preprocessing, DEM/SAR integration, and mask/normalization steps using Sentinel-1, Sentinel-2, and elevation data producing stackable chips for training robust Earth observation models. The Final dataset contains 11 bands including seven bands from sentinel2, Elevation, slope from Copernicus DEM, RTC (Radiometrically terrain corrected)VV and VH layers from Sentinel-1.

---

## 📦 Installation

Install directly from PyPI:

Make sure to pre-install: pip install planetary-computer, pip install awscli



```bash
pip install dl4eo
```

Or install locally in development mode:

```bash
git clone https://github.com/your-username/dl4eo.git
cd dl4eo
pip install -e .
```

---

## 🚀 Quick Start

```python
import dl4eo

dl4eo.generate_dataset(
    base_dir="/your/output_directory",
    aoi_shapefile_dir="/path/to/aoi/folder",  # folder containing one or more AOI shapefiles
    feature_shapefile="/path/to/lakes.shp",   # shapefile used for AOI box creation & visualization
    date_range="2020-08-01/2020-08-30",
    box_size_m=1280,                          # image chip extent (default = 2560)
    cloud_cover=15                            # optional: set max cloud cover % (default = 20)
)
```

---

## 🧠 What It Does

The pipeline consists of the following automated stages:

1. **Download Sentinel-2 imagery** via STAC API (cloud-cover filtered)
2. **Preprocess S2**: resampling and band stacking
3. **AOI Box Generation**: intersects lakes/AOI to create image chips
4. **DEM Integration**: clips and resamples elevation data (e.g., SRTM or TanDEM-X)
5. **Download Sentinel-1 (SAR)** from ASF, clips and stacks VV/VH
6. **Mask Generation** using the provided lake shapefile
7. **Data Normalization** across the full stack

---

## 📂 Input Requirements

- `aoi_shapefile_dir`: Folder containing one or more AOI `.shp` files
- `feature_shapefile`: A shapefile representing features (e.g., lakes) to extract training samples from
- Valid date range in the format: `"YYYY-MM-DD/YYYY-MM-DD"`

---

## 🧰 Dependencies

Installed automatically:
- `rasterio`, `geopandas`, `shapely`, `matplotlib`
- `pystac-client`, `fiona`, `requests`, `numpy`, `joblib`

---

## 🗃 Output Structure

```
output_dir/
├── images/              # Raw Sentinel-2
├── Resampled/           # 10m resampled bands
├── stack/               # Stacked Sentinel-2 bands
├── DEM/                 # Elevation data
├── GRD/                 # Raw Sentinel-1 GRD
├── GRD_Extracted/       # Extracted by bounding box
├── Clipped_SAR/         # Matched SAR chips
├── stacked_with_sar/    # Combined S2 + SAR + DEM
├── mask/                # Binary masks (from lakes)
├── normalize/           # Final normalized image chips
├── AOI_boxes/           # AOI boxes (GeoJSONs)
└── shapefile/each/      # Individual AOI shapefiles
```

---

## 🧪 Example Use Cases

- Glacial lake mapping and segmentation
- Flood extent extraction
- Multimodal image fusion (S2+S1+DEM)
- Chip-based data generation for training transformers and GANs

---

## 🧑‍💻 Author

Developed by [Saurabh Kaushik](https://scholar.google.com/citations?user=UBGlaXIAAAAJ),  
Postdoctoral Researcher @ University of Arizona  
Earth Observation, Deep Learning, Geo-Foundational Models, Cryosphere

---

## 📜 License

MIT License

---

## ?? Citation

If you use `dl4eo` in your research or publications, please cite it as:
Kaushik, S. (2025). dl4eo: A Python package for multi-source remote sensing data preparation for deep learning.
Python Package Index. https://pypi.org/project/dl4eo/


BibTeX:
```bibtex
@misc{kaushik2025dl4eo,
  author       = {Saurabh Kaushik},
  title        = {{dl4eo: A Python package for multi-source remote sensing data preparation for deep learning}},
  year         = {2025},
  howpublished = {\url{https://pypi.org/project/dl4eo/}},
}
