"""
Pytest configuration file for PGPTracker tests.

This file configures pytest options and fixtures that are available
across all test files.
"""

import pytest


def pytest_addoption(parser):
    """Add custom command line options to pytest."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (integration tests, system checks)"
    )


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically skip slow tests unless --run-slow is specified."""
    if config.getoption("--run-slow"):
        # --run-slow given in cli: do not skip slow tests
        return
    
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)