# src/pgptracker/stage2_analysis/clustering_ML.py
import polars as pl
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from boruta import BorutaPy
from typing import Literal, Tuple, Dict, Any, List

def _prepare_ml_data(
    df_wide_N_D_clr: pl.DataFrame,
    metadata: pl.DataFrame,
    sample_id_col: str,
    target_col: str,
    analysis_type: Literal['classification', 'regression']
) -> Tuple[np.ndarray, np.ndarray, List[str], Dict[int, Any]]:
    """
    Prepares data for scikit-learn from N×D table and metadata.

    1. Joins N×D CLR data with metadata on sample_id_col.
    2. Aligns and filters samples present in both.
    3. Encodes target_col 'y' (if classification).
    4. Returns aligned X (N, D) and y (N,) arrays.
    
    Args:
        df_wide_N_D_clr: N×D (samples × features) CLR-transformed data.
        metadata: Metadata table.
        sample_id_col: Name of the sample ID column (e.g., "Sample").
        target_col: Name of the target variable column in metadata.
        analysis_type: 'classification' or 'regression'.

    Returns:
        Tuple: (X, y, feature_names, target_map)
    """
    if sample_id_col not in df_wide_N_D_clr.columns:
        raise KeyError(f"Sample ID column '{sample_id_col}' not in wide data.")
    if sample_id_col not in metadata.columns:
        raise KeyError(f"Sample ID column '{sample_id_col}' not in metadata.")
    if target_col not in metadata.columns:
        raise KeyError(f"Target column '{target_col}' not in metadata.")

    # Select only the necessary columns from metadata
    md_subset = metadata.select([sample_id_col, target_col])
    
    # Join wide data with metadata
    df_joined = df_wide_N_D_clr.join(md_subset, on=sample_id_col, how="inner")
    
    if df_joined.height == 0:
        raise ValueError(
            f"No matching samples found between data and metadata on column '{sample_id_col}'."
        )

    # Prepare X
    feature_names = [
        c for c in df_wide_N_D_clr.columns if c != sample_id_col
    ]
    X = df_joined.select(feature_names).to_numpy()
    
    # Prepare y
    y_series = df_joined.get_column(target_col)
    target_map = {}

    if analysis_type == 'classification':
        if y_series.dtype not in [pl.Categorical, pl.String]:
            y_series = y_series.cast(pl.String)
            
        # Create integer-encoded 'y' and the map
        categories = y_series.unique().sort().to_list()
        target_map = {i: cat for i, cat in enumerate(categories)}
        cat_map = {cat: i for i, cat in enumerate(categories)}
        
        y = y_series.replace(cat_map).cast(pl.Int64).to_numpy()
        
    elif analysis_type == 'regression':
        if not y_series.dtype.is_numeric():
            raise ValueError(
                f"For 'regression', target '{target_col}' must be numeric, "
                f"but found {y_series.dtype}."
            )
        y = y_series.to_numpy()
        
    return X, y, feature_names, target_map

def run_lasso_cv(
    df_wide_N_D_clr: pl.DataFrame,
    metadata: pl.DataFrame,
    sample_id_col: str,
    target_col: str,
    n_cv: int = 5,
    random_state: int = 42
) -> pl.DataFrame:
    """
    Runs LassoCV regression to find important features.
    """
    X, y, feats, _ = _prepare_ml_data(
        df_wide_N_D_clr, metadata, sample_id_col, 
        target_col, 'regression'
    )
    
    # Ensure n_cv is not greater than n_samples
    n_samples = X.shape[0]
    if n_cv > n_samples:
        print(f"  -> Warning: n_cv ({n_cv}) > n_samples ({n_samples}). Setting n_cv=n_samples.")
        n_cv = n_samples
        
    lasso = LassoCV(cv=n_cv, random_state=random_state, n_jobs=-1, max_iter=10000).fit(X, y)
    
    results = pl.DataFrame({
        "Feature": feats,
        "Coefficient": lasso.coef_
    })
    
    # Filter out zero coefficients and sort
    return results.filter(pl.col("Coefficient") != 0).sort(
        "Coefficient", descending=True
    )

def run_random_forest(
    df_wide_N_D_clr: pl.DataFrame,
    metadata: pl.DataFrame,
    sample_id_col: str,
    target_col: str,
    analysis_type: Literal['classification', 'regression'],
    random_state: int = 42
) -> pl.DataFrame:
    """
    Runs Random Forest to find important features.
    """
    X, y, feats, _ = _prepare_ml_data(
        df_wide_N_D_clr, metadata, sample_id_col, 
        target_col, analysis_type
    )

    if analysis_type == 'classification':
        rf = RandomForestClassifier(random_state=random_state, n_estimators=500, n_jobs=-1)
    else:
        rf = RandomForestRegressor(random_state=random_state, n_estimators=500, n_jobs=-1)
        
    rf.fit(X, y)
    
    results = pl.DataFrame({
        "Feature": feats,
        "Importance": rf.feature_importances_
    })
    
    return results.sort("Importance", descending=True)

def run_boruta(
    df_wide_N_D_clr: pl.DataFrame,
    metadata: pl.DataFrame,
    sample_id_col: str,
    target_col: str,
    random_state: int = 42
) -> pl.DataFrame:
    """
    Runs Boruta feature selection (Classification only).
    """
    X, y, feats, _ = _prepare_ml_data(
        df_wide_N_D_clr, metadata, sample_id_col, 
        target_col, 'classification'
    )
    
    rf = RandomForestClassifier(
        random_state=random_state, n_estimators=500, n_jobs=-1
    )
    boruta = BorutaPy(
        estimator=rf, 
        n_estimators='auto', # type: ignore
        random_state=random_state,
        max_iter=100
    )
    
    boruta.fit(X, y)
    
    decision_map = {
        True: "Confirmed",
        False: "Rejected",
        np.nan: "Tentative" # Should not happen if max_iter is reached
    }
    
    results = pl.DataFrame({
        "Feature": feats,
        "Decision_Bool": boruta.support_,
        "Decision_Weak": boruta.support_weak_,
        "Rank": boruta.ranking_
    }).with_columns(
        pl.when(pl.col("Decision_Bool") == True)
        .then(pl.lit("Confirmed"))
        .otherwise(
            pl.when(pl.col("Decision_Weak") == True)
            .then(pl.lit("Tentative"))
            .otherwise(pl.lit("Rejected"))
        )
        .alias("Decision")
    ).select(["Feature", "Decision", "Rank"]).sort("Rank")
    
    return results