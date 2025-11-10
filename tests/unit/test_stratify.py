"""
Unit tests for pgptracker.analysis.stratify logic.

This test suite focuses on the Polars transformations, aggregations,
and calculations (sum/mean) as requested.

Run with: pytest tests/unit/test_stratify.py -v
"""

import pytest
import polars as pl
import polars.testing as pl_testing
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import io
import gzip

# Import all functions from the script provided by the user
from pgptracker.analysis.stratify import (
    identify_sample_columns,
    aggregate_by_tax_level_sample,
    aggregate_by_tax_level_ko,
    join_and_calculate_batched,
    generate_stratified_analysis,
    TAXONOMY_COLS
)
# Mock the dependencies that are imported
from pgptracker.utils import validator
from pgptracker.analysis import unstratified

# --- Fixtures for Mock Data ---

@pytest.fixture
def tax_level():
    """Provides the taxonomic level used in tests."""
    return "Genus"

@pytest.fixture
def ftable_mock(tax_level):
    """
    Mock DataFrame for the merged feature table (ftable).
    Includes:
    - Multiple ASVs per Genus
    - 'None' (unclassified) Genus
    - Zeros in sample columns
    - An extra metadata column
    """
    return pl.DataFrame({
        "ASV_ID": ["A1", "A2", "A3", "A4", "A5"],
        tax_level: ["GenusA", "GenusA", "GenusB", None, "GenusB"],
        "Phylum": ["P1", "P1", "P2", "P1", "P2"], # Another tax col
        "Sample_1": [10, 5, 0, 8, 2],  # A3 has abundance 0
        "Sample_2": [0, 15, 20, 0, 0], # A1, A4, A5 have abundance 0
        "meta_depth": [100, 110, 120, 130, 140] # Extra metadata col
    })

@pytest.fixture
def ko_mock():
    """
    Mock DataFrame for KO predictions.
    Includes:
    - Different ASV_ID column name ('sequence')
    - Zeros in KO columns
    - ASVs not present in ftable_mock (A5 is missing)
    - Metadata column
    """
    return pl.DataFrame({
        "sequence": ["A1", "A2", "A3", "A4"], # A5 is missing
        "ko:K001": [1.0, 2.0, 0.0, 4.0], # A3 has 0
        "ko:K002": [0.0, 3.0, 5.0, 0.0], # A1, A4 have 0
        "metadata_NSTI": [0.1, 0.2, 0.3, 0.4] # Extra metadata col
    })

@pytest.fixture
def pathways_mock():
    """
    Mock DataFrame for KO-to-PGPT mapping.
    Includes:
    - One KO mapping to multiple PGPTs
    """
    return pl.DataFrame({
        "KO": ["ko:K001", "ko:K002", "ko:K002"],
        "Lv3": ["PGPT_A", "PGPT_A", "PGPT_B"] # K002 maps to PGPT_A and PGPT_B
    })

# --- Tests for identify_sample_columns ---

class TestIdentifySampleColumns:

    # Mock find_asv_column to simplify testing
    @patch('pgptracker.analysis.stratify.find_asv_column', return_value="ASV_ID")
    def test_default_behavior(self, mock_find, ftable_mock):
        """
        Tests default behavior: should return all columns that are not
        ASV_ID or recognized TAXONOMY_COLS.
        """
        # Note: 'meta_depth' is included, as expected by the code's logic
        expected = ["Sample_1", "Sample_2", "meta_depth"]
        result = identify_sample_columns(ftable_mock)
        assert sorted(result) == sorted(expected)

    @patch('pgptracker.analysis.stratify.find_asv_column', return_value="ASV_ID")
    def test_with_sample_prefix(self, mock_find, ftable_mock):
        """Tests prefix-based filtering."""
        expected = ["Sample_1", "Sample_2"]
        result = identify_sample_columns(ftable_mock, sample_prefix="Sample_")
        assert sorted(result) == sorted(expected)

    @patch('pgptracker.analysis.stratify.find_asv_column', return_value="ASV_ID")
    def test_with_exclude_cols(self, mock_find, ftable_mock):
        """Tests metadata exclusion-based filtering."""
        expected = ["Sample_1", "Sample_2"]
        result = identify_sample_columns(ftable_mock, exclude_cols=["meta_depth"])
        assert sorted(result) == sorted(expected)

    @patch('pgptracker.analysis.stratify.find_asv_column', return_value="ASV_ID")
    def test_no_samples_found_error(self, mock_find):
        """Tests that it raises ValueError if no sample columns are left."""
        simple_ftable = pl.DataFrame({
            "ASV_ID": ["A1"],
            "Genus": ["GenusA"]
        })
        with pytest.raises(ValueError, match="No sample columns identified"):
            identify_sample_columns(simple_ftable)

# --- Tests for aggregate_by_tax_level_sample (Abundance SUM) ---

class TestAggregateByTaxLevelSample:

    @patch('pgptracker.analysis.stratify.find_asv_column', return_value="ASV_ID")
    def test_aggregation_sum_and_zero_filter(self, mock_find, ftable_mock, tax_level):
        """
        Tests the core logic: unpivot, filter zeros, and SUM abundances.
        Also tests that 'None' taxa are dropped by default.
        """
        sample_cols = ["Sample_1", "Sample_2"]

        # Expected Math:
        # GenusA, Sample_1: 10 (A1) + 5 (A2) = 15
        # GenusA, Sample_2: 15 (A2) (A1=0 is filtered)
        # GenusB, Sample_1: 2 (A5) (A3=0 is filtered)
        # GenusB, Sample_2: 20 (A3) (A5=0 is filtered)
        # Genus=None (A4) is dropped (Sample_1=8, Sample_2=0)
        expected_df = pl.DataFrame({
            tax_level: ["GenusA", "GenusA", "GenusB", "GenusB"],
            "Sample": ["Sample_1", "Sample_2", "Sample_1", "Sample_2"],
            "Total_Tax_Abundance": [15.0, 15.0, 2.0, 20.0]
        }).sort([tax_level, "Sample"])

        result_df = aggregate_by_tax_level_sample(
            ftable_mock,
            tax_level,
            sample_cols,
            keep_unclassified_samples=False # Default
        ).sort([tax_level, "Sample"])

        pl_testing.assert_frame_equal(result_df, expected_df, check_dtypes=False)

    @patch('pgptracker.analysis.stratify.find_asv_column', return_value="ASV_ID")
    def test_keep_unclassified_true(self, mock_find, ftable_mock, tax_level):
        """Tests that 'keep_unclassified_samples=True' renames 'None' taxa."""
        sample_cols = ["Sample_1", "Sample_2"]

        # Expected: Same as before, but includes the 'None' group from A4
        # Genus=None (A4), Sample_1: 8 (A4,S2=0 is filtered)
        expected_df = pl.DataFrame({
            tax_level: ["GenusA", "GenusA", "GenusB", "GenusB", f"Unclassified_{tax_level}"],
            "Sample": ["Sample_1", "Sample_2", "Sample_1", "Sample_2", "Sample_1"],
            "Total_Tax_Abundance": [15.0, 15.0, 2.0, 20.0, 8.0]
        }).sort([tax_level, "Sample"])

        result_df = aggregate_by_tax_level_sample(
            ftable_mock,
            tax_level,
            sample_cols,
            keep_unclassified_samples=True
        ).sort([tax_level, "Sample"])

        pl_testing.assert_frame_equal(result_df, expected_df, check_dtypes=False)

# --- Tests for aggregate_by_tax_level_ko (KO MEAN) ---

class TestAggregateByTaxLevelKO:

    @patch('pgptracker.analysis.stratify.find_asv_column', side_effect=["ASV_ID", "sequence"])
    def test_aggregation_mean_and_mapping(self, mock_find, ftable_mock, ko_mock, tax_level):
        """
        Tests the core logic: KO unpivot, zero filter, taxonomy join, and MEAN.
        Also tests different ASV_ID column names.
        """
        # Expected Math:
        # 1. tax_map (from ftable_mock, unique):
        #    (A1, GenusA), (A2, GenusA), (A3, GenusB), (A4, None), (A5, GenusB)
        # 2. ko_long (from ko_mock, filtered):
        #    (A1, K001, 1.0), (A2, K001, 2.0), (A2, K002, 3.0), (A3, K002, 5.0), (A4, K001, 4.0)
        # 3. ko_with_tax (joined):
        #    (A1, K001, 1.0, GenusA), (A2, K001, 2.0, GenusA), (A2, K002, 3.0, GenusA),
        #    (A3, K002, 5.0, GenusB), (A4, K001, 4.0, None)
        # 4. tax_ko (aggregated with MEAN):
        #    GenusA, K001: mean(1.0, 2.0) = 1.5
        #    GenusA, K002: mean(3.0)      = 3.0
        #    GenusB, K002: mean(5.0)      = 5.0
        #    None,   K001: mean(4.0)      = 4.0
        expected_df = pl.DataFrame({
            tax_level: ["GenusA", "GenusA", "GenusB", None],
            "KO": ["ko:K001", "ko:K002", "ko:K002", "ko:K001"],
            "Avg_Copy_Number": [1.5, 3.0, 5.0, 4.0]
        }).sort([tax_level, "KO"])

        result_df = aggregate_by_tax_level_ko(
            ko_mock,
            ftable_mock,
            tax_level
        ).sort([tax_level, "KO"])

        pl_testing.assert_frame_equal(result_df, expected_df, check_dtypes=False)
        # Check that find_asv_column was called for both DFs
        assert mock_find.call_count == 2
        mock_find.assert_has_calls([call(ftable_mock), call(ko_mock)])


# --- Tests for join_and_calculate_batched (Final Math) ---

class TestJoinAndCalculateBatched:

    @patch('gzip.open')
    @patch('io.TextIOWrapper')
    def test_batch_math_and_io(self, mock_text_wrapper, mock_gzip, tax_level, pathways_mock):
        """
        Tests the final calculation (Abundance * Copy_Number) and aggregation.
        Mocks the file I/O to capture the string output.
        """
        # --- Arrange ---
        # 1. Use the *expected* outputs from the previous tests as inputs
        tax_abun_input = pl.DataFrame({
            tax_level: ["GenusA", "GenusA", "GenusB", "GenusB", None],
            "Sample": ["Sample_1", "Sample_2", "Sample_1", "Sample_2", "Sample_1"],
            "Total_Tax_Abundance": [15.0, 15.0, 2.0, 20.0, 8.0] # Includes 'None' group
        })
        tax_ko_input = pl.DataFrame({
            tax_level: ["GenusA", "GenusA", "GenusB", None],
            "KO": ["ko:K001", "ko:K002", "ko:K002", "ko:K001"],
            "Avg_Copy_Number": [1.5, 3.0, 5.0, 4.0]
        })
        pgpt_level = "Lv3"
        mock_output_path = Path("/mock/output.tsv.gz")

        # Mock the file writing part
        mock_file_stream = io.StringIO()
        mock_text_wrapper.return_value.__enter__.return_value = mock_file_stream
        mock_gzip.return_value.__enter__.return_value = MagicMock() # gzip mock

        # --- Act ---
        join_and_calculate_batched(
            tax_abun_input,
            tax_ko_input,
            pathways_mock,
            mock_output_path,
            tax_level,
            pgpt_level
        )

        # --- Assert ---
        # Get the string that was written to the file
        output_string = mock_file_stream.getvalue()
        
        # Read the string back into Polars for validation
        result_df = pl.read_csv(io.BytesIO(output_string.encode('utf-8')), separator='\t') \
            .sort([tax_level, pgpt_level, "Sample"])

        # --- Expected Math ---
        # ko_pgpt_map (tax_ko JOIN pathways):
        #   (GenusA, K001, 1.5, PGPT_A)
        #   (GenusA, K002, 3.0, PGPT_A)
        #   (GenusA, K002, 3.0, PGPT_B)
        #   (GenusB, K002, 5.0, PGPT_A)
        #   (GenusB, K002, 5.0, PGPT_B)
        #   (None, K001, 4.0, PGPT_A)
        #
        # Batch 1 'GenusA' (Abun: S1=15, S2=15):
        #   S1, K001, PGPT_A: 15 * 1.5 = 22.5
        #   S1, K002, PGPT_A: 15 * 3.0 = 45.0
        #   S1, K002, PGPT_B: 15 * 3.0 = 45.0
        #   S2, K001, PGPT_A: 15 * 1.5 = 22.5
        #   S2, K002, PGPT_A: 15 * 3.0 = 45.0
        #   S2, K002, PGPT_B: 15 * 3.0 = 45.0
        #   Grouped:
        #     (GenusA, PGPT_A, S1): 22.5 + 45.0 = 67.5
        #     (GenusA, PGPT_B, S1): 45.0
        #     (GenusA, PGPT_A, S2): 22.5 + 45.0 = 67.5
        #     (GenusA, PGPT_B, S2): 45.0
        #
        # Batch 2 'GenusB' (Abun: S1=2, S2=20):
        #   S1, K002, PGPT_A: 2 * 5.0 = 10.0
        #   S1, K002, PGPT_B: 2 * 5.0 = 10.0
        #   S2, K002, PGPT_A: 20 * 5.0 = 100.0
        #   S2, K002, PGPT_B: 20 * 5.0 = 100.0
        #   Grouped:
        #     (GenusB, PGPT_A, S1): 10.0
        #     (GenusB, PGPT_B, S1): 10.0
        #     (GenusB, PGPT_A, S2): 100.0
        #     (GenusB, PGPT_B, S2): 100.0
        #
        # Batch 3 'None' (Abun: S1=8):
        #   S1, K001, PGPT_A: 8 * 4.0 = 32.0
        #   Grouped:
        #     (None, PGPT_A, S1): 32.0
        #
        expected_df = pl.DataFrame({
            tax_level: ["GenusA", "GenusA", "GenusA", "GenusA", "GenusB", "GenusB", "GenusB", "GenusB", None],
            pgpt_level: ["PGPT_A", "PGPT_A", "PGPT_B", "PGPT_B", "PGPT_A", "PGPT_A", "PGPT_B", "PGPT_B", "PGPT_A"],
            "Sample": ["Sample_1", "Sample_2", "Sample_1", "Sample_2", "Sample_1", "Sample_2", "Sample_1", "Sample_2", "Sample_1"],
            "Total_PGPT_Abundance": [67.5, 67.5, 45.0, 45.0, 10.0, 100.0, 10.0, 100.0, 32.0]
        }).sort([tax_level, pgpt_level, "Sample"])

        pl_testing.assert_frame_equal(result_df, expected_df, check_dtypes=False)
        
        # Check that header was written (first line of string)
        assert output_string.startswith(f"{tax_level}\t{pgpt_level}\tSample\tTotal_PGPT_Abundance")

# --- Tests for generate_stratified_analysis (Orchestrator) ---

class TestGenerateStratifiedAnalysis:

    @patch('pgptracker.analysis.stratify.pl.read_csv')
    @patch('pgptracker.analysis.stratify.load_pathways_db')
    @patch('pgptracker.analysis.stratify.identify_sample_columns')
    @patch('pgptracker.analysis.stratify.join_and_calculate_batched')
    def test_orchestrator_flow(
        self,
        mock_join_calc,
        mock_identify_cols,
        mock_load_pathways,
        mock_pl_read_csv,
        ftable_mock, # Use fixtures
        ko_mock,     # Use fixtures
        pathways_mock,
        tax_level,
        tmp_path
    ):
        """
        Tests the main orchestrator function.
        Mocks I/O and the final batch function to spy on the data flow.
        """
        # --- Arrange ---
        # Mock file I/O
        mock_ko_df_dropped = ko_mock.drop("metadata_NSTI")
        mock_ko_mock_with_drop = MagicMock(spec=pl.DataFrame)
        mock_ko_mock_with_drop.drop.return_value = mock_ko_df_dropped
        mock_ko_mock_with_drop.columns = ko_mock.columns
        
        # Create a mutable list for the side_effect
        read_csv_returns = [
            ftable_mock, # First call (ftable)
            mock_ko_mock_with_drop  # Second call (ko)
        ]
        mock_pl_read_csv.side_effect = read_csv_returns

        mock_load_pathways.return_value = pathways_mock
        mock_identify_cols.return_value = ["Sample_1", "Sample_2"]
        
        # Mock final sorting I/O
        mock_sorted_df = MagicMock(spec=pl.DataFrame)
        # Append to the mutable list, not the side_effect iterator
        read_csv_returns.append(mock_sorted_df) # Third call (sorting)

        # Add a fourth mock for the snippet preview read
        mock_snippet_df = MagicMock(spec=pl.DataFrame)
        read_csv_returns.append(mock_snippet_df) # Fourth call (snippet)
        
        # Paths
        ftable_path = tmp_path / "ftable.tsv"
        ko_path = tmp_path / "ko.tsv.gz"
        output_dir = tmp_path / "output"
        pgpt_level = "Lv3"

        # --- Act ---
        result_path = generate_stratified_analysis(
            ftable_path,
            ko_path,
            output_dir,
            tax_level,
            pgpt_level,
            sample_prefix=None, # Explicitly pass
            exclude_cols=None,  # Explicitly pass
            keep_unclassified=False # Explicitly pass
        )

        # --- Assert ---
        # Check output path
        assert result_path == output_dir / f"{tax_level.lower()}_stratified_pgpt.tsv.gz"

        # Check I/O calls
        mock_pl_read_csv.assert_has_calls([
            call(ftable_path, separator='\t', has_header=True, comment_prefix='#'),
            call(ko_path, separator='\t', has_header=True),
            call(result_path, separator='\t') # For sorting
        ])
        mock_load_pathways.assert_called_with(pgpt_level=pgpt_level)
        
        # Check metadata was dropped from KO df
        mock_ko_mock_with_drop.drop.assert_called_once()
        assert "metadata_NSTI" in mock_ko_mock_with_drop.drop.call_args[0][0]

        # Check that the final batch function was called
        mock_join_calc.assert_called_once()
        
        # Spy on the arguments passed to the batch function
        call_args = mock_join_calc.call_args[0]
        
        # 1. Check tax_abun arg
        tax_abun_arg = call_args[0]
        # (Check a known value from TestAggregateByTaxLevelSample)
        expected_abun = 15.0 # GenusA, Sample_1
        assert tax_abun_arg.filter(pl.col(tax_level) == "GenusA") \
                           .filter(pl.col("Sample") == "Sample_1") \
                           .select("Total_Tax_Abundance").item() == expected_abun

        # 2. Check tax_ko arg
        tax_ko_arg = call_args[1]
        # (Check a known value from TestAggregateByTaxLevelKO)
        expected_ko_mean = 1.5 # GenusA, K001
        assert tax_ko_arg.filter(pl.col(tax_level) == "GenusA") \
                         .filter(pl.col("KO") == "ko:K001") \
                         .select("Avg_Copy_Number").item() == expected_ko_mean

        # 3. Check other args
        pl_testing.assert_frame_equal(call_args[2], pathways_mock)
        assert call_args[3] == result_path
        assert call_args[4] == tax_level
        assert call_args[5] == pgpt_level