"""
PGPTracker Entry Point.

This module allows PGPTracker to be executed as:
    python -m pgptracker [command] [options]
    
It simply delegates to the CLI main function.
"""
import sys
from pgptracker.cli import main

if __name__ == "__main__":
    sys.exit(main())