"""
Setup script for Greta Core Engine.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="greta-core",
    version="0.1.0",
    author="Greta Team",
    author_email="",
    description="Automated Data Analysis Library with Genetic Algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "deap>=1.3.0",
        "openpyxl>=3.0.0",  # For Excel support
        "dask>=2022.0.0",
        "dask[dataframe]>=2022.0.0",
        "dowhy>=0.10.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "pytest-cov"],
    },
)