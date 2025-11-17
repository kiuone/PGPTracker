# src/pgptracker/gui/__init__.py

"""
PGPTracker Stage 2 Data Explorer GUI.

A modular Dash application for interactive exploration of CLR-transformed
feature tables and metadata from the PGPTracker pipeline.
"""

from pgptracker.gui.app import app, server, run_app

__all__ = ["app", "server", "run_app"]
