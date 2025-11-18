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
    Handles .gz compressed files automatically.

    Args:
        file_path: Path to TSV file (.tsv or .tsv.gz)

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
    Handles .tsv, .csv, .txt, and .gz compressed files.

    Args:
        uploaded_file: Streamlit UploadedFile object

    Returns:
        Polars DataFrame
    """
    # Determine separator based on file extension
    filename = uploaded_file.name.lower()
    separator = "," if filename.endswith(".csv") else "\t"

    return pl.read_csv(
        uploaded_file,
        separator=separator,
        infer_schema_length=None,
        null_values=["NA", "nan", "null", ""]
    )


def detect_table_format(df: pl.DataFrame) -> str:
    """
    Detect if table is Wide (N×D) or Long/Stratified format.

    Args:
        df: Polars DataFrame

    Returns:
        "wide" or "long"
    """
    # Long format indicators: has Sample + Abundance + (Taxonomy/PGPT/Feature)
    cols = df.columns
    cols_lower = [c.lower() for c in cols]

    has_sample = any('sample' in c for c in cols_lower)
    has_abundance = any(c in cols_lower for c in ['abundance', 'count', 'value'])
    has_taxonomy = any(c in cols_lower for c in ['taxonomy', 'taxon', 'feature', 'pgpt', 'function'])

    if has_sample and has_abundance and has_taxonomy:
        return "long"
    else:
        return "wide"


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
        "sample_id", "SAMPLE", "#SampleID", "sample_name"
    ]

    for candidate in common_patterns:
        if candidate in columns:
            return candidate

    return None


def merge_data(
    df_clr: pl.DataFrame,
    df_metadata: pl.DataFrame,
    metadata_sample_col: str
) -> Tuple[pl.DataFrame, List[str], List[str], str]:
    """
    Merge CLR data with metadata, handling both wide and long formats.

    Args:
        df_clr: CLR-transformed data (wide or long format)
        df_metadata: Metadata DataFrame
        metadata_sample_col: Name of sample ID column in metadata

    Returns:
        Tuple of (merged_df, metadata_columns, feature_columns, format_type)
    """
    # Detect format
    format_type = detect_table_format(df_clr)

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

    # Extract column lists based on format
    metadata_cols = [col for col in df_metadata.columns if col != "Sample"]

    if format_type == "long":
        # For long format, feature_cols are categorical columns (Taxonomy, PGPT, etc.)
        feature_cols = [col for col in df_clr.columns
                       if col not in ["Sample", "Abundance", "abundance", "Count", "count", "Value", "value"]]
    else:
        # For wide format, feature_cols are all numeric columns except Sample
        feature_cols = [col for col in df_clr.columns if col != "Sample"]

    return df_merged, metadata_cols, feature_cols, format_type

