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
    log_level = logging.INFO if args.verbose else logging.WARNING
    
    logging.basicConfig(
        level=log_level, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        force=True)
    
    logger.info("Starting Stage 2 Pipeline: Analysis & Modeling")
    
    # 0. Setup Paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_table = Path(args.input_table)
    input_metadata = Path(args.metadata)
    
    if not input_table.exists() or not input_metadata.exists():
        logger.error(f"Input files not found: {input_table} or {input_metadata}")
        sys.exit(1)

    # --- STEP 1: NORMALIZATION (CLR) ---
    logger.info(f"Step 1: Applying CLR Normalization on {input_table.name}...")
    
    try:
        # apply_clr returns paths to both RAW (NxD) and CLR (NxD) tables
        clr_outputs = apply_clr(
            input_path=input_table,
            input_format=args.input_format, 
            output_dir=output_dir / "normalization",
            base_name="data",
            wide_orientation=args.orientation, # 'D_N' or 'N_D' from CLI
            wide_id_col=args.feature_col_name, # e.g., 'Lv3' or 'FeatureID'
            keep_feature_cols_separate=True
        )
        
        # Load the NxD RAW table for Alpha Diversity (counts)
        raw_path = clr_outputs['raw_wide_N_D']
        df_raw = pl.read_csv(raw_path, separator="\t")
        
        # Load the NxD CLR table for Beta Diversity/Stats (normalized)
        clr_path = clr_outputs['clr_wide_N_D']
        df_clr = pl.read_csv(clr_path, separator="\t")

        # Load metadata 
        df_meta = pl.read_csv(input_metadata, separator="\t", infer_schema_length=10000)
        
        # Normalize metadata ID column to 'Sample' to match our pipeline standard
        if args.metadata_id_col != "Sample":
            if args.metadata_id_col in df_meta.columns:
                logger.info(f"Renaming metadata column '{args.metadata_id_col}' to 'Sample' for compatibility.")
                df_meta = df_meta.rename({args.metadata_id_col: "Sample"})
            else:
                if "Sample" not in df_meta.columns:
                    logger.error(f"Metadata ID column '{args.metadata_id_col}' not found in metadata file.")
                    sys.exit(1)
        
        logger.info("Normalization complete.")
        
    except Exception as e:
        logger.error(f"Normalization failed: {e}")
        sys.exit(1)

    # --- STEP 2: DIVERSITY & ORDINATION ---
    logger.info("Step 2: Calculating Diversity & Ordination...")
    
    div_dir = output_dir / "diversity"
    div_dir.mkdir(exist_ok=True)
    
    try:
        # 2a. Alpha Diversity (Uses RAW counts)
        if args.group_col in df_meta.columns:
            logger.info("  -> Calculating Alpha Diversity...")
            alpha_res = calculate_alpha_diversity(
                df_raw, 'Sample', metrics=['observed_features', 'shannon', 'pielou_e', 'simpson']
            )
            alpha_res.write_csv(div_dir / "alpha_diversity.tsv", separator="\t")
            
            plot_alpha_diversity(
                alpha_res, df_meta, 'Sample', args.group_col, output_dir=div_dir
            )

        # 2b. Beta Diversity (Uses CLR - Aitchison)
        logger.info("  -> Calculating Beta Diversity (Aitchison)...")
        dm_aitchison = calculate_beta_diversity(df_clr, 'Sample', 'aitchison')
        
        # 2c. Ordination (PCA & t-SNE on CLR)
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
            plot_ordination(scores_pca, df_meta, 'Sample', args.group_col, df_loadings=loadings_pca,
                            title="PCA Biplot (Aitchison)", output_dir=div_dir, base_name="pca_plot")
            plot_ordination(scores_tsne, df_meta, 'Sample', args.group_col, 
                            x_col="tSNE1", y_col="tSNE2",
                            title="t-SNE", output_dir=div_dir, base_name="tsne_plot")
            
            # 2d. PERMANOVA
            logger.info(f"  -> Running PERMANOVA ({args.group_col})...")
            perm_res = permanova_test(dm_aitchison, df_meta, 'Sample', f"~{args.group_col}")
            
            with open(div_dir / "permanova_results.txt", "w") as f:
                f.write(str(perm_res))

    except Exception as e:
        logger.error(f"Diversity/Ordination failed: {e}")

    gc.collect()

    # --- STEP 3: STATISTICAL TESTING ---
    if args.run_stats and args.group_col in df_meta.columns:
        logger.info(f"Step 3: Statistical Testing (Group: {args.group_col})...")
        stats_dir = output_dir / "statistics"
        stats_dir.mkdir(exist_ok=True)
        
        try:
            # Check number of groups to decide test
            groups = df_meta.get_column(args.group_col).unique().to_list()
            n_groups = len(groups)
            
            stats_res = None
            
            if n_groups == 2:
                logger.info(f"  -> 2 Groups detected ({groups}). Running Mann-Whitney U...")
                stats_res = mann_whitney_u_test(
                    df_clr, df_meta, 'Sample', 'Feature', args.group_col, 'Abundance',
                    group_1=str(groups[0]), group_2=str(groups[1])
                )
            elif n_groups > 2:
                logger.info(f"  -> {n_groups} Groups detected. Running Kruskal-Wallis...")
                stats_res = kruskal_wallis_test(
                    df_clr, df_meta, 'Sample', 'Feature', args.group_col, 'Abundance'
                )
            else:
                logger.warning("  -> Less than 2 groups found. Skipping statistics.")

            if stats_res is not None:
                # FDR Correction
                stats_res = stats_res.with_columns(
                    fdr_correction(stats_res['p_value']).alias("q_value")
                )
                
                stats_res.write_csv(stats_dir / "differential_abundance_results.tsv", separator="\t")
                
                # Plots
                plot_volcano(stats_res, p_val_col="q_value", output_dir=stats_dir)
                
                # Heatmap (Top 50 significant)
                top_feats = stats_res.sort("q_value").head(50)["Feature"].to_list()
                if top_feats:
                    df_heatmap = df_clr.select(["Sample"] + top_feats)
                    plot_heatmap(
                        df_heatmap, df_meta, 'Sample', args.group_col, 
                        output_dir=stats_dir, base_name="heatmap_significant"
                    )

        except Exception as e:
            logger.error(f"Statistics failed: {e}")

    gc.collect()

    # --- STEP 4: MACHINE LEARNING ---
    if args.run_ml and args.target_col in df_meta.columns:
        logger.info(f"Step 4: Machine Learning (Target: {args.target_col})...")
        ml_dir = output_dir / "machine_learning"
        ml_dir.mkdir(exist_ok=True)
        
        try:
            # Random Forest
            rf_res = run_random_forest(
                df_clr, df_meta, 'Sample', args.target_col, 
                analysis_type=args.ml_type
            )
            rf_res.write_csv(ml_dir / "random_forest_importance.tsv", separator="\t")
            plot_feature_importance(rf_res, output_dir=ml_dir, base_name="rf_importance")
            
            # Lasso (Regression Only)
            if args.ml_type == 'regression':
                lasso_res = run_lasso_cv(
                    df_clr, df_meta, 'Sample', args.target_col
                )
                lasso_res.write_csv(ml_dir / "lasso_coefficients.tsv", separator="\t")
                plot_feature_importance(lasso_res, title="Lasso Coefficients", 
                                      output_dir=ml_dir, base_name="lasso_importance")
                                      
            # Boruta (Classification Only)
            if args.ml_type == 'classification':
                boruta_res = run_boruta(
                    df_clr, df_meta, 'Sample', args.target_col
                )
                boruta_res.write_csv(ml_dir / "boruta_selection.tsv", separator="\t")

        except Exception as e:
            logger.error(f"Machine Learning failed: {e}")

    logger.info("Stage 2 Pipeline Completed Successfully.")