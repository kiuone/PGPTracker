#!/usr/bin/env python3
"""
Unit tests for CLR normalization in PGPTracker.

Tests compositional data transformation properties:
1. CLR sum per sample = 0 (geometric mean centering)
2. Zeros are replaced before transformation
3. Wide and long formats handled correctly

Author: Vivian Mello

runs with: pytest tests/unit/test_clr_normalize.py -v
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from pgptracker.stage2_analysis.clr_normalize import detect_tbl_format


class TestCLRWideFormat:

    def test_clr_wide_basic(self):
        """Test CLR on simple wide format data."""
        df = pl.DataFrame({
            'PGPT_ID': ['NITROGEN_FIX', 'PHOSPHATE_SOL', 'SIDEROPHORE'],
            'Sample_A': [10.0, 20.0, 30.0],
            'Sample_B': [5.0, 15.0, 25.0]})
        
        result = detect_tbl_format(df, format='wide')
        
        # Check structure preserved
        assert result.columns == df.columns
        assert len(result) == len(df)
        assert result['PGPT_ID'].to_list() == df['PGPT_ID'].to_list()
    
    def test_clr_wide_sum_zero(self):
        """CLR values per sample must sum to ~0 (geometric mean centering)."""
        df = pl.DataFrame({
            'PGPT_ID': ['PGPT_1', 'PGPT_2', 'PGPT_3'],
            'Sample_A': [10.0, 20.0, 30.0],
            'Sample_B': [100.0, 200.0, 300.0]
        })
        
        result = detect_tbl_format(df, format='wide')
        
        # Sum of CLR values per sample should be ~0
        for col in ['Sample_A', 'Sample_B']:
            col_sum = result[col].sum()
            assert abs(col_sum) < 1e-10, f"{col} sum = {col_sum}, expected ~0"
    
    def test_clr_wide_handles_zeros(self):
        """CLR should handle zeros through multiplicative replacement."""
        df = pl.DataFrame({
            'PGPT_ID': ['PGPT_1', 'PGPT_2', 'PGPT_3'],
            'Sample_A': [10.0, 0.0, 30.0],  # Zero in PGPT_2
            'Sample_B': [5.0, 15.0, 0.0]    # Zero in PGPT_3
        })
        
        result = detect_tbl_format(df, format='wide')
        
        # No NaN or inf values should exist
        for col in ['Sample_A', 'Sample_B']:
            assert not result[col].is_nan().any()
            assert not result[col].is_infinite().any()
        
        # CLR sum still ~0
        for col in ['Sample_A', 'Sample_B']:
            assert abs(result[col].sum()) < 1e-10
    
    def test_clr_wide_negative_positive_values(self):
        """CLR produces both negative and positive values (log-ratios)."""
        df = pl.DataFrame({
            'PGPT_ID': ['PGPT_1', 'PGPT_2', 'PGPT_3'],
            'Sample_A': [10.0, 20.0, 30.0]
        })
        
        result = detect_tbl_format(df, format='wide')
        
        values = result['Sample_A'].to_list()
        assert any(v < 0 for v in values), "Should have negative values"
        assert any(v > 0 for v in values), "Should have positive values"


class TestCLRLongFormat:
    def test_clr_long_basic(self):
        """Test CLR on stratified long format data."""
        df = pl.DataFrame({
            'Order': ['Bacillales', 'Bacillales', 'Pseudomonadales', 'Pseudomonadales'],
            'Lv3': ['NITROGEN_FIX', 'PHOSPHATE_SOL', 'NITROGEN_FIX', 'PHOSPHATE_SOL'],
            'Sample': ['Sample_A', 'Sample_A', 'Sample_A', 'Sample_A'],
            'Total_PGPT_Abundance': [10.0, 20.0, 30.0, 40.0]
        })
        
        result = detect_tbl_format(df, format='long')
        
        # Check structure
        assert 'CLR_Abundance' in result.columns
        assert 'Total_PGPT_Abundance' not in result.columns
        assert len(result) == len(df)
    
    def test_clr_long_sum_zero_per_sample(self):
        """CLR values per sample must sum to ~0."""
        df = pl.DataFrame({
            'Genus': ['Bacillus', 'Pseudomonas', 'Bacillus', 'Pseudomonas'],
            'Lv3': ['NITROGEN_FIX', 'NITROGEN_FIX', 'PHOSPHATE_SOL', 'PHOSPHATE_SOL'],
            'Sample': ['Sample_A', 'Sample_A', 'Sample_A', 'Sample_A'],
            'Total_PGPT_Abundance': [10.0, 20.0, 15.0, 25.0]
        })
        
        result = detect_tbl_format(df, format='long')
        
        # Group by sample and check sum
        for sample in result['Sample'].unique():
            sample_data = result.filter(pl.col('Sample') == sample)
            clr_sum = sample_data['CLR_Abundance'].sum()
            assert abs(clr_sum) < 1e-10, f"Sample {sample} CLR sum = {clr_sum}"
    
    def test_clr_long_handles_zeros(self):
        """CLR should handle zeros in long format."""
        df = pl.DataFrame({
            'Genus': ['Bacillus', 'Pseudomonas'],
            'Lv3': ['NITROGEN_FIX', 'NITROGEN_FIX'],
            'Sample': ['Sample_A', 'Sample_A'],
            'Total_PGPT_Abundance': [10.0, 0.0]  # Zero abundance
        })
        
        result = detect_tbl_format(df, format='long')
        
        # No NaN or inf
        assert not result['CLR_Abundance'].is_nan().any()
        assert not result['CLR_Abundance'].is_infinite().any()
    
    def test_clr_long_multiple_samples(self):
        """Test CLR across multiple samples."""
        df = pl.DataFrame({
            'Genus': ['Bacillus', 'Bacillus', 'Pseudomonas', 'Pseudomonas'],
            'Lv3': ['NITROGEN_FIX', 'NITROGEN_FIX', 'NITROGEN_FIX', 'NITROGEN_FIX'],
            'Sample': ['Sample_A', 'Sample_B', 'Sample_A', 'Sample_B'],
            'Total_PGPT_Abundance': [10.0, 15.0, 20.0, 25.0]
        })
        
        result = detect_tbl_format(df, format='long')
        
        # Each sample independently should sum to ~0
        for sample in ['Sample_A', 'Sample_B']:
            sample_data = result.filter(pl.col('Sample') == sample)
            clr_sum = sample_data['CLR_Abundance'].sum()
            assert abs(clr_sum) < 1e-10


class TestCLRFormatDetection:
    """Tests for automatic format detection."""
    
    def test_detects_wide_format(self):
        """Should detect wide format (no 'Sample' column)."""
        df = pl.DataFrame({
            'PGPT_ID': ['PGPT_1', 'PGPT_2'],
            'Sample_A': [10.0, 20.0],
            'Sample_B': [30.0, 40.0]
        })
        
        result = detect_tbl_format(df, format='wide')
        
        # Wide format: no 'CLR_Abundance' column
        assert 'CLR_Abundance' not in result.columns
        assert 'Sample_A' in result.columns
        assert 'Sample_B' in result.columns
    
    def test_detects_long_format(self):
        """Should detect long format ('Sample' column exists)."""
        df = pl.DataFrame({
            'Genus': ['Bacillus'],
            'Lv3': ['NITROGEN_FIX'],
            'Sample': ['Sample_A'],
            'Total_PGPT_Abundance': [10.0]
        })
        
        result = detect_tbl_format(df, format='long')
        
        # Long format: has 'CLR_Abundance' column
        assert 'CLR_Abundance' in result.columns
        assert 'Sample' in result.columns


class TestCLREdgeCases:
    """Edge cases and error conditions."""
    
    def test_all_zeros_in_sample(self):
        """Handle sample with all zeros (should fill with small values)."""
        df = pl.DataFrame({
            'PGPT_ID': ['PGPT_1', 'PGPT_2'],
            'Sample_A': [0.0, 0.0],  # All zeros
            'Sample_B': [10.0, 20.0]
        })
        
        result = detect_tbl_format(df, format='wide')
        
        # Should not crash and produce valid CLR values
        assert not result['Sample_A'].is_nan().any()
        assert not result['Sample_B'].is_nan().any()
    
    def test_single_feature(self):
        """Handle single feature (CLR should be 0)."""
        df = pl.DataFrame({
            'PGPT_ID': ['PGPT_1'],
            'Sample_A': [100.0]
        })
        
        result = detect_tbl_format(df, format='wide')
        
        # Single feature: CLR = log(x / geom_mean) = log(x / x) = 0
        assert abs(result['Sample_A'][0]) < 1e-10
    
    def test_preserves_feature_order(self):
        """Feature order should be preserved."""
        df = pl.DataFrame({
            'PGPT_ID': ['Z_PGPT', 'A_PGPT', 'M_PGPT'],
            'Sample_A': [10.0, 20.0, 30.0]
        })
        
        result = detect_tbl_format(df, format='wide')
        
        assert result['PGPT_ID'].to_list() == ['Z_PGPT', 'A_PGPT', 'M_PGPT']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])