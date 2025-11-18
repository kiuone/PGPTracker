"""
PGPTracker - Plant Growth-Promoting Traits Analysis Pipeline
Setup configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip() 
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="pgptracker",

    use_scm_version=True, 
    
    author="Vivian Mello",
    author_email="vmellomasc@gmail.com",
    description="Integration of soil metagenomic data for correlation of microbial markers with plant biochemical indicators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kiuone/PGPTracker",
    package_dir={'': 'src'},
    packages=find_packages(where='src', exclude=["tests", "tests.*"]),

    include_package_data=True, 
    
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=4.0",
            "pytest-mock>=3.12",
            "build",
            "twine",
        ],
    },
    entry_points={
        "console_scripts": [
            "pgptracker=pgptracker.cli.cli:main",
        ],
    },

    package_data={
        "pgptracker": ["databases/*.txt",
                       "environments/*.yml"],
    },
)