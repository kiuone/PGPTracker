"""
PGPTracker resource setup utilities.

Validates external dependencies.
"""

import shutil


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
