"""
PGPTracker resource setup utilities.

Downloads reference databases and validates external dependencies.
"""

import shutil
import tarfile
from pathlib import Path

import requests
from tqdm import tqdm


def check_dependencies() -> None:
    """
    Validate external binaries are in PATH.

    Raises:
        RuntimeError: If any dependency is missing
    """
    required = {
        "run_sepp.py": "SEPP (install via: conda install -c bioconda sepp)",
        "gappa": "GAPPA (install via: conda install -c bioconda gappa)",
        "Rscript": "R (install via: conda install -c conda-forge r-base r-castor r-ape)"
    }

    missing = []
    for binary, install_msg in required.items():
        if not shutil.which(binary):
            missing.append(f"  - {binary}: {install_msg}")

    if missing:
        raise RuntimeError(
            "Missing required dependencies:\n" + "\n".join(missing) +
            "\n\nPlease install via conda/mamba before running PGPTracker."
        )


def download_database(output_dir: Path, force: bool = False) -> Path:
    """
    Download PICRUSt2 prokaryotic reference database.

    Streams tarball from official FTP and extracts prokaryotic/ directory containing:
    - pro_ref.fna (reference genomes)
    - pro_ref.tre (reference tree)
    - pro_ref.hmm (HMM profiles)
    - ko.txt.gz (KO abundance table)

    Args:
        output_dir: Target directory for extracted database
        force: Overwrite existing database if True

    Returns:
        Path to prokaryotic database directory

    Raises:
        RuntimeError: If download fails or extraction incomplete
    """
    output_dir = Path(output_dir)
    db_dir = output_dir / "prokaryotic"

    # Skip if database already exists
    if db_dir.exists() and not force:
        required_files = ["pro_ref.fna", "pro_ref.tre", "pro_ref.hmm", "ko.txt.gz"]
        if all((db_dir / f).exists() for f in required_files):
            print(f"Database already exists at {db_dir}")
            return db_dir
        print("Incomplete database detected, re-downloading...")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # PICRUSt2 reference database URL
    url = "https://ftp.microbio.me/pub/picrust2_data/prokaryotic/2.5.2/prokaryotic.tar.gz"
    tarball_path = output_dir / "prokaryotic.tar.gz"

    # Download with progress bar
    print(f"Downloading PICRUSt2 reference database from {url}")
    print("This is a large file (~10GB) and may take several minutes...")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(tarball_path, 'wb') as f, tqdm(
        total=total_size,
        unit='B',
        unit_scale=True,
        desc="Downloading"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    # Extract tarball
    print(f"Extracting database to {output_dir}")
    with tarfile.open(tarball_path, 'r:gz') as tar:
        tar.extractall(path=output_dir)

    # Clean up tarball
    tarball_path.unlink()

    # Validate extraction
    required_files = ["pro_ref.fna", "pro_ref.tre", "pro_ref.hmm", "ko.txt.gz"]
    missing = [f for f in required_files if not (db_dir / f).exists()]

    if missing:
        raise RuntimeError(
            f"Database extraction incomplete. Missing files: {missing}\n"
            "Try re-running with --force flag."
        )

    print(f"Database successfully installed at {db_dir}")
    return db_dir
