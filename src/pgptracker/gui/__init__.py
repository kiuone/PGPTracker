"""
PGPTracker Stage 2 Data Explorer GUI.

A streamlit-based application for interactive exploration of CLR-transformed
feature tables and metadata from the PGPTracker pipeline.
"""

import subprocess
import sys
from pathlib import Path


def run_app(results_dir=None, port=8501):
    """
    Launch the Streamlit GUI.

    Args:
        results_dir: Optional path (str or Path) to results directory for auto-loading
        port: Port to run Streamlit server on (default: 8501)
    """
    # Get path to app.py
    app_path = Path(__file__).parent / "app.py"

    # Build streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--server.headless", "true"
    ]

    # Add results directory if provided
    if results_dir:
        # Convert to Path if string
        if isinstance(results_dir, str):
            results_dir = Path(results_dir)
        cmd.append("--")  # Streamlit separator for script arguments
        cmd.append(str(results_dir))

    # Run streamlit
    subprocess.run(cmd)


__all__ = ["run_app"]
