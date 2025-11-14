#!/usr/bin/env python3
"""
Unit tests for CLR normalization in PGPTracker.
Updated for apply_clr which takes file paths and returns a Dict[str, Path].

Tests compositional data transformation properties:
1. CLR sum per sample = 0 (geometric mean centering)
2. Zeros are replaced before transformation
3. Wide and long formats are correctly pivoted and transformed.

Author: Vivian Mello

runs with: pytest tests/unit/test_clr_normalize.py -v
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

# Módulo sendo testado
from pgptracker.stage2_analysis.clr_normalize import apply_clr


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Cria um diretório de saída para os testes."""
    out_dir = tmp_path / "clr_outputs"
    out_dir.mkdir()
    return out_dir


class TestCLRWideFormat:
    """Tests for 'wide' (unstratified) format."""

    def test_clr_wide_basic(self, output_dir: Path):
        """Test CLR on simple wide format data."""
        df = pl.DataFrame({
            'PGPT_ID': ['NITROGEN_FIX', 'PHOSPHATE_SOL', 'SIDEROPHORE'],
            'Sample_A': [10.0, 20.0, 30.0],
            'Sample_B': [5.0, 15.0, 25.0]})
        
        base_name = "wide_basic.tsv"
        input_path = output_dir / base_name
        df.write_csv(input_path, separator="\t")
        
        outputs = apply_clr(input_path, 'wide', output_dir, base_name)
        
        # Check if the correct keys and files are returned
        assert 'raw_wide' in outputs
        assert 'clr_wide' in outputs
        assert outputs['raw_wide'].exists()
        assert outputs['clr_wide'].exists()
        
        result = pl.read_csv(outputs['clr_wide'], separator="\t")
        
        # Check structure preserved
        assert result.columns == df.columns
        assert len(result) == len(df)
        assert result['PGPT_ID'].to_list() == df['PGPT_ID'].to_list()

    def test_clr_wide_sum_zero(self, output_dir: Path):
        """CLR values per sample must sum to ~0 (geometric mean centering)."""
        df = pl.DataFrame({
            'PGPT_ID': ['PGPT_1', 'PGPT_2', 'PGPT_3'],
            'Sample_A': [10.0, 20.0, 30.0],
            'Sample_B': [100.0, 200.0, 300.0]
        })
        
        base_name = "wide_sum_zero.tsv"
        input_path = output_dir / base_name
        df.write_csv(input_path, separator="\t")
        
        outputs = apply_clr(input_path, 'wide', output_dir, base_name)
        result = pl.read_csv(outputs['clr_wide'], separator="\t")
        
        # Sum of CLR values per sample should be ~0
        for col in ['Sample_A', 'Sample_B']:
            col_sum = result[col].sum()
            assert abs(col_sum) < 1e-10, f"{col} sum = {col_sum}, expected ~0"

    def test_clr_wide_handles_zeros(self, output_dir: Path):
        """CLR should handle zeros through multiplicative replacement."""
        df = pl.DataFrame({
            'PGPT_ID': ['PGPT_1', 'PGPT_2', 'PGPT_3'],
            'Sample_A': [10.0, 0.0, 30.0],  # Zero in PGPT_2
            'Sample_B': [5.0, 15.0, 0.0]    # Zero in PGPT_3
        })
        
        base_name = "wide_zeros.tsv"
        input_path = output_dir / base_name
        df.write_csv(input_path, separator="\t")

        outputs = apply_clr(input_path, 'wide', output_dir, base_name)
        result = pl.read_csv(outputs['clr_wide'], separator="\t")
        
        # No NaN or inf values should exist
        for col in ['Sample_A', 'Sample_B']:
            assert not result[col].is_nan().any()
            assert not result[col].is_infinite().any()
        
        # CLR sum still ~0
        for col in ['Sample_A', 'Sample_B']:
            assert abs(result[col].sum()) < 1e-10

    def test_clr_wide_negative_positive_values(self, output_dir: Path):
        """CLR produces both negative and positive values (log-ratios)."""
        df = pl.DataFrame({
            'PGPT_ID': ['PGPT_1', 'PGPT_2', 'PGPT_3'],
            'Sample_A': [10.0, 20.0, 30.0]
        })

        base_name = "wide_neg_pos.tsv"
        input_path = output_dir / base_name
        df.write_csv(input_path, separator="\t")
        
        outputs = apply_clr(input_path, 'wide', output_dir, base_name)
        result = pl.read_csv(outputs['clr_wide'], separator="\t")
        
        values = result['Sample_A'].to_list()
        assert any(v < 0 for v in values), "Should have negative values"
        assert any(v > 0 for v in values), "Should have positive values"


class TestCLRLongFormat:
    """Tests for 'long' (stratified) format and its pivot-to-wide output."""
    
    def test_clr_long_returns_all_outputs(self, output_dir: Path):
        """Test CLR on stratified long format data."""
        df = pl.DataFrame({
            'Order': ['Bacillales', 'Bacillales', 'Pseudomonadales', 'Pseudomonadales'],
            'Lv3': ['NITROGEN_FIX', 'PHOSPHATE_SOL', 'NITROGEN_FIX', 'PHOSPHATE_SOL'],
            'Sample': ['Sample_A', 'Sample_A', 'Sample_A', 'Sample_A'],
            'Total_PGPT_Abundance': [10.0, 20.0, 30.0, 40.0]
        })
        
        base_name = "long_basic.tsv"
        input_path = output_dir / base_name
        df.write_csv(input_path, separator="\t")

        # Use 'long' format, default sample/value columns
        outputs = apply_clr(input_path, 'long', output_dir, base_name)
        
        # Check if all keys are returned
        assert 'raw_long' in outputs
        assert 'raw_wide' in outputs
        assert 'clr_wide' in outputs
        assert outputs['raw_long'].exists()
        assert outputs['raw_wide'].exists()
        assert outputs['clr_wide'].exists()
        
        result_wide = pl.read_csv(outputs['clr_wide'], separator="\t")
        
        # Check wide structure (pivoted correctly)
        assert 'Sample_A' in result_wide.columns
        assert 'Order' in result_wide.columns
        assert 'Lv3' in result_wide.columns
        assert len(result_wide) == 4

    def test_clr_long_sum_zero_per_sample(self, output_dir: Path):
        """CLR values per sample must sum to ~0 in the wide output."""
        df = pl.DataFrame({
            'Genus': ['Bacillus', 'Pseudomonas', 'Bacillus', 'Pseudomonas'],
            'Lv3': ['NITROGEN_FIX', 'NITROGEN_FIX', 'PHOSPHATE_SOL', 'PHOSPHATE_SOL'],
            'Sample': ['Sample_A', 'Sample_A', 'Sample_A', 'Sample_A'],
            'Total_PGPT_Abundance': [10.0, 20.0, 15.0, 25.0]
        })
        
        base_name = "long_sum_zero.tsv"
        input_path = output_dir / base_name
        df.write_csv(input_path, separator="\t")

        outputs = apply_clr(input_path, 'long', output_dir, base_name)
        result_wide = pl.read_csv(outputs['clr_wide'], separator="\t")
        
        # Test wide output sum (sum sample column)
        for col in ['Sample_A']:
            clr_sum = result_wide[col].sum()
            assert abs(clr_sum) < 1e-10, f"Wide format {col} sum = {clr_sum}"
    
    def test_clr_long_handles_zeros(self, output_dir: Path):
        """CLR should handle zeros in long format (in the wide output)."""
        df = pl.DataFrame({
            'Genus': ['Bacillus', 'Pseudomonas'],
            'Lv3': ['NITROGEN_FIX', 'NITROGEN_FIX'],
            'Sample': ['Sample_A', 'Sample_A'],
            'Total_PGPT_Abundance': [10.0, 0.0]  # Zero abundance
        })
        
        base_name = "long_zeros.tsv"
        input_path = output_dir / base_name
        df.write_csv(input_path, separator="\t")
        
        outputs = apply_clr(input_path, 'long', output_dir, base_name)
        result_wide = pl.read_csv(outputs['clr_wide'], separator="\t")
        
        # Test wide output for NaN/inf
        assert not result_wide['Sample_A'].is_nan().any()
        assert not result_wide['Sample_A'].is_infinite().any()
    
    def test_clr_long_multiple_samples(self, output_dir: Path):
        """Test CLR across multiple samples (pivoted correctly)."""
        df = pl.DataFrame({
            'Genus': ['Bacillus', 'Bacillus', 'Pseudomonas', 'Pseudomonas'],
            'Lv3': ['NITROGEN_FIX', 'NITROGEN_FIX', 'NITROGEN_FIX', 'NITROGEN_FIX'],
            'Sample': ['Sample_A', 'Sample_B', 'Sample_A', 'Sample_B'],
            'Total_PGPT_Abundance': [10.0, 15.0, 20.0, 25.0]
        })
        
        base_name = "long_multi_sample.tsv"
        input_path = output_dir / base_name
        df.write_csv(input_path, separator="\t")

        outputs = apply_clr(input_path, 'long', output_dir, base_name)
        result_wide = pl.read_csv(outputs['clr_wide'], separator="\t")
        
        # Test wide output (column sum)
        for sample in ['Sample_A', 'Sample_B']:
            assert sample in result_wide.columns
            clr_sum = result_wide[sample].sum()
            assert abs(clr_sum) < 1e-10, f"Wide format {sample} sum failed"


class TestCLREdgeCases:
    """Edge cases and error conditions."""
    
    def test_all_zeros_in_sample(self, output_dir: Path):
        """Handle sample with all zeros (CLR should be 0)."""
        df = pl.DataFrame({
            'PGPT_ID': ['PGPT_1', 'PGPT_2'],
            'Sample_A': [0.0, 0.0],   # All zeros
            'Sample_B': [10.0, 20.0]
        })
        
        base_name = "edge_all_zeros.tsv"
        input_path = output_dir / base_name
        df.write_csv(input_path, separator="\t")
        
        outputs = apply_clr(input_path, 'wide', output_dir, base_name)
        result = pl.read_csv(outputs['clr_wide'], separator="\t")
        
        # Should not crash and produce valid CLR values
        assert not result['Sample_A'].is_nan().any()
        assert not result['Sample_B'].is_nan().any()
        
        # All-zero sample CLR sum must be 0
        assert abs(result['Sample_A'].sum()) < 1e-10
        # Valid sample sum must be ~0
        assert abs(result['Sample_B'].sum()) < 1e-10

    def test_single_feature(self, output_dir: Path):
        """Handle single feature (CLR should be 0)."""
        df = pl.DataFrame({
            'PGPT_ID': ['PGPT_1'],
            'Sample_A': [100.0]
        })
        
        base_name = "edge_single_feature.tsv"
        input_path = output_dir / base_name
        df.write_csv(input_path, separator="\t")

        outputs = apply_clr(input_path, 'wide', output_dir, base_name)
        result = pl.read_csv(outputs['clr_wide'], separator="\t")
        
        # Single feature: CLR = log(x / geom_mean) = log(x / x) = 0
        assert abs(result['Sample_A'][0]) < 1e-10
    
    def test_preserves_feature_order(self, output_dir: Path):
        """Feature order should be preserved in wide format."""
        df = pl.DataFrame({
            'PGPT_ID': ['Z_PGPT', 'A_PGPT', 'M_PGPT'],
            'Sample_A': [10.0, 20.0, 30.0]
        })
        
        base_name = "edge_feature_order.tsv"
        input_path = output_dir / base_name
        df.write_csv(input_path, separator="\t")
        
        outputs = apply_clr(input_path, 'wide', output_dir, base_name)
        result = pl.read_csv(outputs['clr_wide'], separator="\t")
        
        assert result['PGPT_ID'].to_list() == ['Z_PGPT', 'A_PGPT', 'M_PGPT']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])