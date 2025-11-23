#!/usr/bin/env python3
# run_gui.py

"""
Launch the PGPTracker Stage 2 Data Explorer GUI.

Usage:
    python run_gui.py [--port PORT] [--no-debug] [-v|--verbose]

Example:
    python run_gui.py --port 8080 --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pgptracker.gui import run_app


def setup_logging(verbose: bool):
    """
    Configure logging based on verbosity level.

    Args:
        verbose: If True, set to INFO level; otherwise WARNING
    """
    level = logging.INFO if verbose else logging.WARNING

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Suppress overly verbose Dash logs unless in verbose mode
    if not verbose:
        logging.getLogger('dash').setLevel(logging.WARNING)
        logging.getLogger('werkzeug').setLevel(logging.WARNING)


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
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging (INFO level)"
    )

    args = parser.parse_args()

    # Configure logging
    setup_logging(args.verbose)

    if args.verbose:
        print("=" * 60)
        print("PGPTracker Stage 2 Data Explorer")
        print("=" * 60)
        print(f"Starting server on http://0.0.0.0:{args.port}")
        print(f"Debug mode: {not args.no_debug}")
        print(f"Verbose logging: enabled")
        print("Press Ctrl+C to stop the server")
        print("=" * 60)

    run_app(debug=not args.no_debug, port=args.port)


if __name__ == "__main__":
    main()
