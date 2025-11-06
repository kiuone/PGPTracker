"""
Table merging module for PGPTracker.

Handles the complex chain of unzipping, BIOM conversion, and metadata merging.
Uses Polars for memory-efficient, streaming processing of large tables.
"""
import subprocess
import gzip
import shutil
import polars as pl
from pathlib import Path
from pgptracker.utils.env_manager import run_command
from pgptracker.utils.validator import ValidationError
from pgptracker.utils.validator import validate_output_file as _validate_output

def _process_taxonomy_polars(df_lazy: pl.LazyFrame) -> pl.LazyFrame:
    """
    Lazily splits taxonomy column into levels and reorders columns using Polars.
    
    Tasks:
    1. Split 'taxonomy' into Kingdom, Phylum, etc.
    2. Clean prefixes (e.g., 'k__')
    3. Reorder columns to: ASV_ID | Tax | Samples...
    """
    # Define standard taxonomic levels
    tax_levels = ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species']
    
    # 1. Split taxonomy string 'k__Bacteria; p__Firmicutes; ...'
    print("    -> Defining lazy logic for taxonomy splitting...")
    df_lazy = df_lazy.with_columns(
        pl.col('taxonomy').str.split('; ').alias('tax_split')
    )
    
    # 2. Lazily create new columns for each level
    for i, level_name in enumerate(tax_levels):
        df_lazy = df_lazy.with_columns(
            pl.col('tax_split')
            .list.get(i)
            .str.replace(r"^[kpcofgs]__", "") # Clean prefix
            .alias(level_name)
        )
    
    # Replace empty strings (like 'g__') with null
    for level_name in tax_levels:
        df_lazy = df_lazy.with_columns(
            pl.when(pl.col(level_name) == "")
            .then(None)
            .otherwise(pl.col(level_name))
            .alias(level_name)
        )

    # 3. Reorder columns
    # Get sample columns (all columns not in the sets below)
    tax_and_helpers = set(tax_levels) | {'ASV_ID', 'taxonomy', 'confidence', 'tax_split'}
    
    # We must 'collect_schema' to know all column names for reordering
    all_cols = df_lazy.collect_schema().names()
    
    sample_cols = [col for col in all_cols if col not in tax_and_helpers]
    
    # New order: ASV_ID, then taxonomy, then all sample columns
    new_order = ['ASV_ID'] + tax_levels + sample_cols
    
    # Select the final columns in the correct order
    df_lazy = df_lazy.select(new_order)
    
    return df_lazy

def merge_taxonomy_to_table(
    seqtab_norm_gz: Path,
    taxonomy_tsv: Path,
    output_dir: Path,
    save_intermediates: bool = False
) -> Path:
    
    """
    Runs the BIOM conversion, merging, and Polars processing pipeline.

    1. Unzips 'seqtab_norm.tsv.gz'
    2. Converts 'seqtab_norm.tsv' -> 'seqtab_norm.biom'
    3. Merges 'taxonomy.tsv' -> 'feature_table_with_taxonomy.biom'
    4. Converts '...with_taxonomy.biom' -> 'temp_merged_table.tsv'
    5. Processes 'temp_merged_table.tsv' -> 'norm_wt_feature_table.tsv' (using Polars)
    """
    
    # Define caminhos
    work_dir = output_dir / "merge_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    
    seqtab_tsv = work_dir / "seqtab_norm.tsv"
    seqtab_biom = work_dir / "seqtab_norm.biom"
    merged_biom = work_dir / "feature_table_with_taxonomy.biom"
    raw_merged_tsv = work_dir / "temp_merged_table.tsv"
    final_processed_tsv = output_dir / "norm_wt_feature_table.tsv" # Final name

    try:
        # Step 1: Unzip 'seqtab_norm.tsv.gz' (o 'gunzip -c')
        if not raw_merged_tsv.exists():
            print("  -> Unzipping normalized table...")
            with gzip.open(seqtab_norm_gz, 'rb') as f_in:
                with open(seqtab_tsv, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            _validate_output(seqtab_tsv, "gunzip", "unzipped sequence table")

            # Step 2: Converts TSV -> BIOM (o 'biom convert') 
            print("  -> Converting normalized table to BIOM format...")
            cmd_convert1 = [
                "biom", "convert",
                "-i", str(seqtab_tsv),
                "-o", str(seqtab_biom),
                "--table-type=OTU table",
                "--to-hdf5"
            ]
            # 'biom' is in 'pgptracker' environment
            run_command("PGPTracker", cmd_convert1, check=True, capture_output=True)
            _validate_output(seqtab_biom, "biom convert", "normalized BIOM table")

            # Step 3: Adds metadata (o 'biom add-metadata') ---
            print("  -> Merging taxonomy into BIOM table...")
            cmd_merge = [
                "biom", "add-metadata",
                "-i", str(seqtab_biom),
                "-o", str(merged_biom),
                "--observation-metadata-fp", str(taxonomy_tsv),
                "--sc-separated", "taxonomy"
            ]
            run_command("PGPTracker", cmd_merge, check=True, capture_output=True)
            _validate_output(merged_biom, "biom add-metadata", "merged BIOM table")

            # Step 4: Converts BIOM -> TSV (o 'biom convert' final) ---
            print("  -> Converting final BIOM table to TSV...")
            cmd_convert2 = [
                "biom", "convert",
                "-i", str(merged_biom),
                "-o", str(raw_merged_tsv),
                "--to-tsv"
            ]
            run_command("PGPTracker", cmd_convert2, check=True)
            _validate_output(raw_merged_tsv, "biom convert", "raw merged TSV")
        
        # Step 5: Process with Polars
        print("  -> Lazily scanning raw merged table with Polars...")
        # Scan the file (lazy)
        df_lazy = pl.scan_csv(raw_merged_tsv, separator='\t', skip_rows=1)
        
        # Rename the OTU ID column (lazy)
        df_lazy = df_lazy.rename({'#OTUID': 'ASV_ID'})
        
        # Call the lazy processing helper function
        processed_lazy = _process_taxonomy_polars(df_lazy)
        
        # Step 6: Print Snippets
        print("  -> Executing query for data snippets...")
        try:
            # Get 3x3 head
            print("\n--- Data Head (First 3 ASVs, First 3 Columns) ---")
            snippet_head = processed_lazy.head(3).select(pl.all().head(3)).collect()
            with pl.Config(tbl_rows=3, tbl_cols=3, tbl_width_chars=80):
                print(snippet_head)

            # Get 5x3 tail
            print("\n--- Data Tail (Last 5 ASVs, Last 3 Columns) ---")
            snippet_tail = processed_lazy.tail(5).select(pl.all().tail(3)).collect()
            with pl.Config(tbl_rows=5, tbl_cols=3, tbl_width_chars=80):
                print(snippet_tail)
            print("\n")

        except Exception as e:
            print(f"  [Warning] Could not print data snippets: {e}")

        # Step 7: Write final processed table to disk
        print(f"  -> Streaming processed table to: {final_processed_tsv} ...")
        # This is the key:
        # 1. collect(engine="streaming") processes the file in chunks (solves RAM issue)
        # 2. write_csv() saves the result
        processed_lazy.collect(engine="streaming").write_csv(final_processed_tsv, separator='\t')
        
        _validate_output(final_processed_tsv, "Polars Processing", "final processed table")

    except (subprocess.CalledProcessError, ValidationError, gzip.BadGzipFile) as e:
        print(f"  [ERROR] BIOM merging pipeline failed: {e}")
        raise RuntimeError("BIOM merging pipeline failed.") from e
    except Exception as e:
        # Catch Polars errors
        print(f"  [ERROR] Post-processing (Polars) failed: {e}")
        raise RuntimeError("Taxonomy processing failed.") from e
    
    # Cleanup
    if not save_intermediates:
        print("  -> Cleaning up intermediate merge files...")
        shutil.rmtree(work_dir)

    print(f"  -> Final merged table ready: {final_processed_tsv}")
    return final_processed_tsv