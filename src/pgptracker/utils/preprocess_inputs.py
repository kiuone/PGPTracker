#!/usr/bin/env python3
"""
Pre-processes input files for PGPTracker.

This script performs two critical functions:
1.  Generates a unique, short ID (e.g., "ASV_00001") for each unique ASV sequence.
2.  Creates an "ASV dictionary" (asv_dictionary.tsv) mapping these IDs to
    their full sequence and taxonomic classification for downstream traceability.
3.  Replaces the long ASV sequence strings in the feature table and KO prediction
    table with these efficient short IDs, saving new processed files.

This solves:
-   **Traceability:** The dictionary allows linking aggregated results back to ASVs.
-   **Performance:** Joining on short string IDs is significantly faster and more
    memory-efficient than joining on long sequence strings.

Author: Vivian Mello (with AI assistance)
"""

import polars as pl
import argparse
import gzip
import sys
from pathlib import Path

from pgptracker.utils.validator import find_asv_column


def preprocess_inputs(
    ftable_path: Path,
    ko_path: Path,
    output_dir: Path
):
    """
    Main preprocessing function.

    Generates ASV IDs, creates dictionary, and replaces sequences with IDs
    in both feature table and KO prediction files.

    Args:
        ftable_path: Path to normalized feature table with taxonomy
        ko_path: Path to KO predictions file (gzipped)
        output_dir: Directory to save processed files

    Outputs:
        - asv_dictionary.tsv: ASV_ID -> Sequence -> Taxonomy mapping
        - ftable_processed.tsv: Feature table with ASV_ID instead of sequences
        - ko_processed.tsv.gz: KO table with ASV_ID instead of sequences
    """
    print(f"--- Starting Pre-processing ---")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  -> Loading feature table: {ftable_path.name}")
    ftable = pl.read_csv(ftable_path, separator='\t', has_header=True, comment_prefix='#')

    print(f"  -> Loading KO predictions: {ko_path.name}")
    with gzip.open(ko_path, 'rb') as f:
        content = f.read()
    ko_df = pl.read_csv(content, separator='\t', has_header=True)

    asv_col_ftable = find_asv_column(ftable)
    asv_col_ko = find_asv_column(ko_df)
    print(f"  -> Found ASV column in feature table: '{asv_col_ftable}'")
    print(f"  -> Found ASV column in KO table: '{asv_col_ko}'")

    print("  -> Generating unique ASV IDs...")
    asv_sequences = ftable.select(pl.col(asv_col_ftable)).unique()

    # Create map: ASV_Sequence -> ASV_ID using 7 digits (e.g., ASV_0012257)
    asv_map = asv_sequences.with_row_count(name="row_id").select(
        pl.col(asv_col_ftable),
        pl.format("ASV_{:07d}", pl.col("row_id")).alias("ASV_ID")
    )

    # Create ASV dictionary with taxonomy
    tax_cols = [c for c in ftable.columns if c in ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']]

    asv_dictionary = ftable.select(
        [asv_col_ftable] + tax_cols
    ).unique(
        subset=[asv_col_ftable]
    ).join(
        asv_map, on=asv_col_ftable, how="inner"
    ).select(
        ["ASV_ID", asv_col_ftable] + tax_cols
    )

    dict_path = output_dir / "asv_dictionary.tsv"
    print(f"  -> Saving ASV Dictionary ({len(asv_dictionary)} ASVs) to: {dict_path}")
    asv_dictionary.write_csv(dict_path, separator='\t')

    # Process feature table: replace sequence column with ASV_ID
    print(f"  -> Processing feature table (replacing '{asv_col_ftable}' with 'ASV_ID')...")
    ftable_processed = ftable.join(
        asv_map, on=asv_col_ftable, how="left"
    ).drop(
        asv_col_ftable
    ).select(
        ["ASV_ID"] + [c for c in ftable.columns if c != asv_col_ftable]
    )

    ftable_out_path = output_dir / "ftable_processed.tsv"
    ftable_processed.write_csv(ftable_out_path, separator='\t')
    print(f"  -> Saved processed feature table to: {ftable_out_path.name}")

    # Process KO table: replace sequence column with ASV_ID
    print(f"  -> Processing KO table (replacing '{asv_col_ko}' with 'ASV_ID')...")
    ko_df_processed = ko_df.join(
        asv_map, left_on=asv_col_ko, right_on=asv_col_ftable, how="left"
    ).drop(
        asv_col_ko, asv_col_ftable
    ).select(
        ["ASV_ID"] + [c for c in ko_df.columns if c != asv_col_ko]
    )

    ko_out_path = output_dir / "ko_processed.tsv.gz"
    ko_df_processed.write_csv(ko_out_path, separator='\t', compression="gzip")
    print(f"  -> Saved processed KO table to: {ko_out_path.name}")

    print(f"--- Pre-processing Complete ---")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-process PGPTracker input files to create stable ASV_IDs.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-i", "--input_table",
        type=Path,
        required=True,
        help="Path to the normalized feature table with taxonomy (e.g., norm_wt_feature_table.tsv)"
    )
    parser.add_argument(
        "-k", "--ko_table",
        type=Path,
        required=True,
        help="Path to the KO predictions file (e.g., KO_predicted.tsv.gz)"
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=Path,
        required=True,
        help="Directory to save the processed files and the ASV dictionary."
    )

    args = parser.parse_args()

    preprocess_inputs(
        ftable_path=args.input_table,
        ko_path=args.ko_table,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
