# src/pgptracker/stage2_analysis/pipeline_st2.py

import argparse
import sys
import gc
from pathlib import Path
import polars as pl
import logging

# Import internal modules (Assuming installed via package)
from pgptracker.stage2_analysis.clr_normalize import apply_clr
from pgptracker.stage2_analysis.diversity import (
calculate_alpha_diversity, calculate_beta_diversity, permanova_test)
from pgptracker.stage2_analysis.ordination import run_pca, run_tsne
from pgptracker.stage2_analysis.statistics import (
    kruskal_wallis_test, mann_whitney_u_test, fdr_correction)
from pgptracker.stage2_analysis.clustering_ML import (
    run_random_forest, run_lasso_cv, run_boruta)
from pgptracker.stage2_analysis.visualizations import (
    plot_ordination, plot_alpha_diversity, plot_feature_importance, 
    plot_volcano, plot_heatmap)

# Setup Logger
logger = logging.getLogger(__name__)

def run_stage2_pipeline(args: argparse.Namespace):
    """
    Main orchestrator for Stage 2: Analysis & Statistical Modeling.
    """
    # Configuração de Logging
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', force=True)
    
    logger.info("Starting Stage 2 Pipeline: Analysis & Modeling")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_table = Path(args.input_table)
    input_metadata = Path(args.metadata)
    
    # Setup formats
    plot_formats = args.plot_formats if hasattr(args, 'plot_formats') else ['png', 'pdf']

    if not input_table.exists() or not input_metadata.exists():
        logger.error(f"Input files not found: {input_table} or {input_metadata}")
        sys.exit(1)

    # --- STEP 1: NORMALIZATION (CLR) ---
    logger.info(f"Step 1: Applying CLR Normalization on {input_table.name}...")
    try:
        clr_outputs = apply_clr(
            input_path=input_table,
            input_format=args.input_format,
            output_dir=output_dir / "normalization",
            base_name="data",
            wide_orientation=args.orientation,
            wide_id_col=args.feature_col_name,
            keep_feature_cols_separate=True
        )
        raw_path = clr_outputs['raw_wide_N_D']
        df_raw = pl.read_csv(raw_path, separator="\t")
        clr_path = clr_outputs['clr_wide_N_D']
        df_clr = pl.read_csv(clr_path, separator="\t")
        
        # --- FIX: Robust Metadata Reading ---
        # Increase schema inference length and handle common NA values
        df_meta = pl.read_csv(
            input_metadata, 
            separator="\t", 
            infer_schema_length=10000, 
            null_values=["NA", "nan", "NaN", "null", ""]
        )
        
        # Ensure Sample column exists (rename #SampleID if present)
        if '#SampleID' in df_meta.columns and 'Sample' not in df_meta.columns:
            logger.info("Renaming metadata column '#SampleID' to 'Sample' for compatibility.")
            df_meta = df_meta.rename({'#SampleID': 'Sample'})
            
        logger.info("Normalization complete.")
    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        sys.exit(1)

    # --- STEP 2: DIVERSITY & ORDINATION ---
    logger.info("Step 2: Calculating Diversity & Ordination...")
    div_dir = output_dir / "diversity"
    div_dir.mkdir(exist_ok=True)
    
    # 2a. Alpha Diversity
    if args.group_col in df_meta.columns:
        logger.info("  -> Calculating Alpha Diversity...")
        alpha_res = calculate_alpha_diversity(
            df_raw, 'Sample', metrics=['observed_features', 'shannon', 'pielou_e', 'simpson']
        )
        alpha_res.write_csv(div_dir / "alpha_diversity.tsv", separator="\t")
        
        plot_alpha_diversity(
            alpha_res, df_meta, 'Sample', args.group_col, output_dir=div_dir,
            formats=plot_formats
        )

    # 2b. Beta Diversity
    logger.info("  -> Calculating Beta Diversity (Aitchison)...")
    dm_aitchison = calculate_beta_diversity(df_clr, 'Sample', 'aitchison')
    
    # 2c. Ordination
    logger.info("  -> Running Ordination...")
    scores_pca, loadings_pca, _ = run_pca(df_clr, 'Sample')
    scores_tsne = run_tsne(
        df_clr.drop("Sample").to_numpy(), 
        df_clr["Sample"].to_list(), 
        perplexity=args.tsne_perplexity
    )
    
    scores_pca.write_csv(div_dir / "pca_scores.tsv", separator="\t")
    scores_tsne.write_csv(div_dir / "tsne_scores.tsv", separator="\t")
    
    if args.group_col in df_meta.columns:
        plot_ordination(
            scores_pca, df_meta, 'Sample', args.group_col, 
            df_loadings=loadings_pca,
            title="PCA Biplot (Aitchison)", output_dir=div_dir, base_name="pca_biplot",
            formats=plot_formats
        )
        plot_ordination(
            scores_tsne, df_meta, 'Sample', args.group_col, 
            x_col="tSNE1", y_col="tSNE2",
            title="t-SNE", output_dir=div_dir, base_name="tsne_plot",
            formats=plot_formats
        )
        
        logger.info(f"  -> Running PERMANOVA ({args.group_col})...")
        perm_res = permanova_test(dm_aitchison, df_meta, 'Sample', f"~{args.group_col}")
        with open(div_dir / "permanova_results.txt", "w") as f:
            f.write(str(perm_res))

    gc.collect()

    # --- STEP 3: STATISTICAL TESTING ---
    if args.run_stats and args.group_col in df_meta.columns:
        logger.info(f"Step 3: Statistical Testing (Group: {args.group_col})...")
        stats_dir = output_dir / "statistics"
        stats_dir.mkdir(exist_ok=True)
        
        groups = df_meta.get_column(args.group_col).unique().to_list()
        n_groups = len(groups)
        stats_res = None
        
        if n_groups == 2:
            logger.info(f"  -> 2 Groups detected. Running Mann-Whitney U...")
            stats_res = mann_whitney_u_test(
                df_clr, df_meta, 'Sample', 'Feature', args.group_col, 'Abundance',
                group_1=str(groups[0]), group_2=str(groups[1])
            )
        elif n_groups > 2:
            logger.info(f"  -> {n_groups} Groups detected. Running Kruskal-Wallis...")
            stats_res = kruskal_wallis_test(
                df_clr, df_meta, 'Sample', 'Feature', args.group_col, 'Abundance'
            )

        if stats_res is not None:
            stats_res = stats_res.with_columns(
                fdr_correction(stats_res['p_value']).alias("q_value")
            )
            stats_res.write_csv(stats_dir / "differential_abundance_results.tsv", separator="\t")
            
            plot_volcano(stats_res, p_val_col="q_value", output_dir=stats_dir, formats=plot_formats)
            
            # FIX: Provide a larger candidate pool (top 500) so visualizations.py 
            # can apply the diversity filter effectively.
            candidates = stats_res.filter(pl.col("q_value") < 0.05)
            
            # Fallback: If few significant features, take top 500 by q-value anyway
            if candidates.height < 50:
                 candidates = stats_res.sort("q_value")

            top_feats = candidates["Feature"].to_list()

            if top_feats:
                plot_heatmap(
                    df_clr, df_meta, 'Sample', args.group_col, 
                    feature_list=top_feats, 
                    top_n_features=50, 
                    output_dir=stats_dir, base_name="heatmap_significant",
                    formats=plot_formats)

    gc.collect()

    # --- STEP 4: MACHINE LEARNING ---
    if args.run_ml and args.target_col in df_meta.columns:
        logger.info(f"Step 4: Machine Learning (Target: {args.target_col})...")
        ml_dir = output_dir / "machine_learning"
        ml_dir.mkdir(exist_ok=True)
        stats_dir = output_dir / "statistics"

        df_ml = df_clr 
        stats_path = stats_dir / "differential_abundance_results.tsv"
        if stats_path.exists(): 
            df_stats = pl.read_csv(stats_path, separator="\t")
            significant_features = df_stats.filter(pl.col("q_value") < 0.05).sort("q_value").head(2000)["Feature"].to_list()
            
            if not significant_features:
                logger.warning("No significant features found. Running ML on full data.")
            else:
                logger.info(f"  -> Training ML model using {len(significant_features)} significant features.")
                df_ml = df_clr.select(["Sample"] + significant_features)

        rf_res = run_random_forest(
            df_ml, df_meta, 'Sample', args.target_col, analysis_type=args.ml_type)
        
        rf_res.write_csv(ml_dir / "random_forest_importance.tsv", separator="\t")
        
        plot_feature_importance(rf_res, output_dir=ml_dir, base_name="rf_importance", formats=plot_formats)
        
        # FIX: Remove limit. Send full importance list for diversity filtering.
        top_ml_feats = rf_res["Feature"].to_list()
        
        if top_ml_feats:
                plot_heatmap(
                df_clr, df_meta, 'Sample', args.group_col,
                feature_list=top_ml_feats, 
                top_n_features=20, 
                output_dir=ml_dir, base_name="heatmap_ml_top20",
                formats=plot_formats)

        if args.ml_type == 'regression':
            lasso_res = run_lasso_cv(df_clr, df_meta, 'Sample', args.target_col)
            lasso_res.write_csv(ml_dir / "lasso_coefficients.tsv", separator="\t")
            plot_feature_importance(lasso_res, title="Lasso Coefficients", 
                                    output_dir=ml_dir, base_name="lasso_importance", formats=plot_formats)
                                    
        if args.ml_type == 'classification':
            boruta_res = run_boruta(df_clr, df_meta, 'Sample', args.target_col)
            boruta_res.write_csv(ml_dir / "boruta_selection.tsv", separator="\t")

    logger.info("Stage 2 Pipeline Completed Successfully.")