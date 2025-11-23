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


def detect_table_format(df: pl.DataFrame, filename: Optional[str] = None) -> str:
    """
    Detect if table is Wide (N×D), Wide-Stratified (N×D with Taxon|PGPT), or Long/Stratified format.

    Detection strategy (in order of priority):
    1. Filename patterns (if provided) - following clr_normalize.py conventions
    2. Column structure analysis (fallback)
    3. Composite Key detection (| in column names)

    Args:
        df: Polars DataFrame
        filename: Optional filename to use for pattern matching

    Returns:
        "wide", "wide-stratified", or "long"
    """
    # Priority 1: Filename-based detection (most reliable)
    if filename:
        filename_lower = filename.lower()

        # Patterns from clr_normalize.py:
        # raw_long_*, *_stratified_*, *stratified* = long format
        # raw_wide_*, clr_wide_*, *unstratified* = wide format
        if any(pattern in filename_lower for pattern in ['raw_long', 'stratified']):
            # Check if it's wide-stratified (filename has 'stratified' but format is wide)
            cols = df.columns
            cols_lower = [c.lower() for c in cols]
            has_sample = any('sample' in c for c in cols_lower)
            has_abundance = any(c in cols_lower for c in ['abundance', 'count', 'value', 'total_pgpt_abundance'])

            if not (has_sample and has_abundance):
                # It's stratified in wide format (pivoted)
                # Check for composite keys
                has_pipe = any('|' in str(col) for col in df.columns if col != 'Sample')
                return "wide-stratified" if has_pipe else "wide"
            else:
                return "long"
        elif any(pattern in filename_lower for pattern in ['raw_wide', 'clr_wide', 'unstratified']):
            # Check for composite keys even in "unstratified" files
            has_pipe = any('|' in str(col) for col in df.columns if col != 'Sample')
            return "wide-stratified" if has_pipe else "wide"

    # Priority 2: Column-based detection (fallback)
    # Long format indicators: has Sample + Abundance + (Taxonomy/PGPT/Feature)
    cols = df.columns
    cols_lower = [c.lower() for c in cols]

    has_sample = any('sample' in c for c in cols_lower)
    has_abundance = any(c in cols_lower for c in ['abundance', 'count', 'value', 'total_pgpt_abundance'])
    has_taxonomy = any(c in cols_lower for c in ['taxonomy', 'taxon', 'feature', 'pgpt', 'function', 'family', 'lv3'])

    # Long format: has explicit Sample, Abundance/Value, and Feature identifier columns
    if has_sample and has_abundance and has_taxonomy:
        return "long"
    else:
        # Wide format: check for composite keys (Taxon|PGPT)
        has_pipe = any('|' in str(col) for col in df.columns if col.lower() != 'sample')
        return "wide-stratified" if has_pipe else "wide"


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


def is_clr_transformed(df: pl.DataFrame, format_type: str, filename: Optional[str] = None) -> bool:
    """
    Detect if data is CLR-transformed or raw counts.

    Detection strategy:
    1. Filename contains 'clr' -> CLR
    2. Has negative values -> CLR (raw counts are never negative)
    3. All values are small floats (< 100) and has negatives -> CLR
    4. Otherwise -> Raw

    Args:
        df: DataFrame to check
        format_type: One of 'wide', 'wide-stratified', or 'long'
        filename: Optional filename for pattern matching

    Returns:
        True if CLR-transformed, False if raw counts
    """
    # Check filename first (most reliable)
    if filename and 'clr' in filename.lower():
        return True
    if filename and 'raw' in filename.lower():
        return False

    # Get numeric columns to check
    if format_type == "long":
        # For long format, check the abundance column
        abundance_cols = [col for col in df.columns
                         if col.lower() in ['abundance', 'count', 'value', 'total_pgpt_abundance']]
        if not abundance_cols:
            return False  # Can't determine, assume raw
        check_col = abundance_cols[0]
        values = df[check_col].drop_nulls()
    else:
        # For wide/wide-stratified, check feature columns (skip Sample column)
        feature_cols = [col for col in df.columns if col.lower() != 'sample']
        if not feature_cols:
            return False
        # Sample first numeric column
        check_col = feature_cols[0]
        values = df[check_col].drop_nulls()

    # If we have values to check
    if len(values) > 0:
        # Check for negative values (CLR can be negative, raw counts cannot)
        has_negative = (values < 0).any()
        if has_negative:
            return True

        # Check typical value range
        max_val = values.max()
        min_val = values.min()

        # CLR typically ranges from -10 to +10, raw counts can be thousands
        if max_val < 100 and min_val >= -100:
            # Likely CLR (small range)
            return True
        elif max_val > 1000:
            # Likely raw counts (large values)
            return False

    # Default: assume raw if can't determine
    return False


def merge_data(
    df_clr: pl.DataFrame,
    df_metadata: pl.DataFrame,
    metadata_sample_col: str,
    clr_filename: Optional[str] = None
) -> Tuple[pl.DataFrame, List[str], List[str], str, bool]:
    """
    Merge CLR data with metadata, handling both wide and long formats.

    Args:
        df_clr: CLR-transformed or raw data (wide or long format)
        df_metadata: Metadata DataFrame
        metadata_sample_col: Name of sample ID column in metadata
        clr_filename: Optional filename for better format detection

    Returns:
        Tuple of (merged_df, metadata_columns, feature_columns, format_type, is_clr)
    """
    # Detect format
    format_type = detect_table_format(df_clr, filename=clr_filename)

    # Detect if data is CLR-transformed
    is_clr = is_clr_transformed(df_clr, format_type, filename=clr_filename)

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
                       if col not in ["Sample", "Abundance", "abundance", "Count", "count", "Value", "value",
                                     "Total_PGPT_Abundance", "total_pgpt_abundance"]]
    else:
        # For wide format, feature_cols are all numeric columns except Sample
        feature_cols = [col for col in df_clr.columns if col != "Sample"]

    return df_merged, metadata_cols, feature_cols, format_type, is_clr

