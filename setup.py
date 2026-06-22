from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dl4eo",
    version="0.5.4",
    description=(
        "Deep Learning for Earth Observation — "
        "automated training-dataset builder for EO segmentation tasks"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Saurabh Kaushik",
    author_email="saurabh21.kaushik@gmail.com",
    url="https://github.com/Sk-2103/dl4eo",
    packages=find_packages(include=["dl4eo", "dl4eo.*"]),
    install_requires=[
        "numpy>=1.22",
        "rasterio>=1.3",
        "geopandas>=0.12",
        "shapely>=1.8",
        "matplotlib>=3.5",
        "joblib>=1.1",
        "pystac-client>=0.6",
        "planetary-computer>=0.4",
        "fiona>=1.8",
        "requests>=2.28",
        "scipy>=1.8",
    ],
    extras_require={
        "train": [
            "torch>=2.0",
            "lightning>=2.0",
            "segmentation-models-pytorch>=0.3",
            "timm>=0.9",
            "torchmetrics>=1.0",
        ],
        "torchgeo": [
            "torchgeo>=0.5",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "dl4eo-run=dl4eo.pipeline:generate_dataset",
        ]
    },
)
