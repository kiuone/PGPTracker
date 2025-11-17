#!/usr/bin/env python3
# run_gui.py

"""
Launch the PGPTracker Stage 2 Data Explorer GUI.

Usage:
    python run_gui.py [--port PORT] [--no-debug]

Example:
    python run_gui.py --port 8080
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pgptracker.gui import run_app


def main():
    parser = argparse.ArgumentParser(
        description="Launch PGPTracker Stage 2 Data Explorer GUI"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port number for the server (default: 8050)"
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Disable debug mode"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PGPTracker Stage 2 Data Explorer")
    print("=" * 60)
    print(f"Starting server on http://0.0.0.0:{args.port}")
    print("Press Ctrl+C to stop the server")
    print("=" * 60)

    run_app(debug=not args.no_debug, port=args.port)


if __name__ == "__main__":
    main()
