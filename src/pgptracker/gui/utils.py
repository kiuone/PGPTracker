"""
Utility functions for data loading and processing.
"""

import streamlit as st
import polars as pl
from pathlib import Path
from typing import Optional, Tuple, List


@st.cache_data
def load_tsv_file(file_path: Path) -> pl.DataFrame:
    """
    Load a TSV file with caching to prevent memory issues.

    Args:
        file_path: Path to TSV file

    Returns:
        Polars DataFrame
    """
    return pl.read_csv(
        file_path,
        separator="\t",
        infer_schema_length=None,  # Scan entire file for correct types
        null_values=["NA", "nan", "null", ""]
    )


@st.cache_data
def load_uploaded_file(uploaded_file) -> pl.DataFrame:
    """
    Load an uploaded file with caching.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Polars DataFrame
    """
    return pl.read_csv(
        uploaded_file,
        separator="\t",
        infer_schema_length=None,
        null_values=["NA", "nan", "null", ""]
    )


def auto_detect_sample_column(columns: List[str]) -> Optional[str]:
    """
    Auto-detect sample ID column from common patterns.

    Args:
        columns: List of column names

    Returns:
        Detected column name or None
    """
    common_patterns = [
        "Sample", "SampleID", "sample", "sampleID",
        "sample_id", "SAMPLE", "#SampleID"
    ]

    for candidate in common_patterns:
        if candidate in columns:
            return candidate

    return None


def merge_data(
    df_clr: pl.DataFrame,
    df_metadata: pl.DataFrame,
    metadata_sample_col: str
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    """
    Merge CLR data with metadata, harmonizing sample column names.

    Args:
        df_clr: CLR-transformed data
        df_metadata: Metadata DataFrame
        metadata_sample_col: Name of sample ID column in metadata

    Returns:
        Tuple of (merged_df, metadata_columns, feature_columns)
    """
    # Auto-detect sample column in CLR data
    clr_sample_col = auto_detect_sample_column(df_clr.columns)
    if not clr_sample_col:
        # Fallback: use first column
        clr_sample_col = df_clr.columns[0]

    # Harmonize column names to "Sample"
    if metadata_sample_col != "Sample":
        df_metadata = df_metadata.rename({metadata_sample_col: "Sample"})

    if clr_sample_col != "Sample":
        df_clr = df_clr.rename({clr_sample_col: "Sample"})

    # Merge on "Sample"
    df_merged = df_clr.join(df_metadata, on="Sample", how="inner")

    # Extract column lists
    metadata_cols = [col for col in df_metadata.columns if col != "Sample"]
    feature_cols = [col for col in df_clr.columns if col != "Sample"]

    return df_merged, metadata_cols, feature_cols
