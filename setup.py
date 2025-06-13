from setuptools import setup, find_packages

# Load README.md content
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dl4eo",
    version="0.2.0",
    description="Deep Learning for Earth Observation - Data Preparation Pipeline",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Saurabh Kaushik",
    author_email="saurabh21.kaushik@gmail.com",  # 
    url="https://github.com/yourusername/dl4eo",  # 
    packages=find_packages(include=["dl4eo", "dl4eo.*"]),
    install_requires=[
        "numpy", "rasterio", "geopandas", "shapely", "matplotlib",
        "joblib", "pystac-client", "fiona", "requests"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS"
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "dl4eo-run=dl4eo.pipeline:generate_dataset"
        ]
    }
)
