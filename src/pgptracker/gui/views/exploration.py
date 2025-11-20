"""
Data exploration view with interactive visualizations.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import numpy as np


def render():
    """Render the data exploration view."""

    # Check if data is loaded
    if not st.session_state.get('data_loaded', False):
        st.warning("⚠️ Please load data first in the 'Upload Data' tab.")
        return

    st.markdown("## 🔍 Interactive Data Exploration")

    # Get data from session state
    df_merged = st.session_state.df_merged
    metadata_cols = st.session_state.metadata_cols
    feature_cols = st.session_state.feature_cols
    format_type = st.session_state.get('format_type', 'wide')
    is_clr = st.session_state.get('is_clr', False)  # CLR IN -> CLR OUT guarantee

    # Normalize format type: "wide-stratified" is treated as "wide" for visualization logic
    # but we show the distinction to the user
    is_stratified = format_type == "wide-stratified"
    format_type_normalized = "wide" if format_type in ["wide", "wide-stratified"] else format_type

    # For WIDE-STRATIFIED: Extract separate lists of Taxons and PGPTs
    taxon_list = []
    pgpt_list = []
    if is_stratified:
        for col in feature_cols:
            if '|' in col:
                parts = col.split('|')
                taxon = parts[0]
                pgpt = '|'.join(parts[1:])  # In case there are multiple pipes
                if taxon not in taxon_list:
                    taxon_list.append(taxon)
                if pgpt not in pgpt_list:
                    pgpt_list.append(pgpt)
        taxon_list = sorted(taxon_list)
        pgpt_list = sorted(pgpt_list)

        # Safety check: ensure we have valid taxons and pgpts
        if not taxon_list or not pgpt_list:
            st.error(f"⚠️ **Invalid WIDE-STRATIFIED format**: Expected 'Taxon|PGPT' columns but found {len(taxon_list)} unique Taxons and {len(pgpt_list)} unique PGPTs")
            st.info("Check that your column names follow the pattern: `TaxonName|PGPT_Name` (e.g., `Bacteroidaceae|NITROGEN_FIXATION`)")
            st.stop()

    # Show format info
    if format_type_normalized == "long":
        st.info("📊 **LONG/STRATIFIED Format** detected - Using Abundance column for visualization")
    elif is_stratified:
        st.info(f"📊 **WIDE-STRATIFIED Format** - {len(taxon_list)} unique Taxons × {len(pgpt_list)} unique PGPTs = {len(feature_cols)} features")
    else:
        st.info("📊 **WIDE Format** detected - Each column is a separate feature")

    # Control panel
    st.markdown("### 📊 Plot Controls")

    col1, col2 = st.columns(2)

    with col1:
        group_col = st.selectbox(
            "Group By (Metadata Column):",
            options=metadata_cols,
            help="Select a metadata column to group samples"
        )

    with col2:
        if format_type_normalized == "long":
            # For long format, let user select taxonomic/functional column
            feature_col = st.selectbox(
                "Stratification Column:",
                options=feature_cols,
                help="Select the taxonomic or functional stratification column (e.g., Taxonomy, PGPT)"
            )
        elif is_stratified:
            # For wide-stratified: Let user choose to view by Taxon OR PGPT
            view_by = st.radio(
                "View by:",
                ["Taxon", "PGPT"],
                horizontal=True,
                help="Aggregate data by Taxon or by PGPT"
            )

            if view_by == "Taxon":
                selected_entity = st.selectbox(
                    "Select Taxon:",
                    options=taxon_list,
                    help="Choose a taxonomic group"
                )
                # This will be used to aggregate all columns starting with this taxon
                feature_col = selected_entity
            else:  # PGPT
                selected_entity = st.selectbox(
                    "Select PGPT:",
                    options=pgpt_list,
                    help="Choose a functional category"
                )
                # This will be used to aggregate all columns ending with this PGPT
                feature_col = selected_entity
        else:
            # Regular wide format
            feature_col = st.selectbox(
                "Feature (Abundance):",
                options=feature_cols,
                help="Select a feature to visualize"
            )

    # Visualization tabs
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "📦 Boxplot",
        "📈 Scatter Plot",
        "📋 Summary Table",
        "📊 Statistical Tests"
    ])

    with viz_tab1:
        if format_type_normalized == "long":
            st.markdown(f"### Abundance Distribution by **{group_col}**")
            st.caption(f"Stratified by: {feature_col}")

            # For long format, use Abundance column
            # Detect abundance column name (PGPTracker uses Total_PGPT_Abundance)
            abundance_col = None
            for col in df_merged.columns:
                if col.lower() in ['abundance', 'count', 'value', 'total_pgpt_abundance']:
                    abundance_col = col
                    break

            if abundance_col:
                # Create boxplot with Abundance on y-axis
                fig = px.box(
                    df_merged,
                    x=group_col,
                    y=abundance_col,
                    color=group_col,
                    points="all",
                    title=f"Abundance Distribution by {group_col}"
                )
            else:
                st.error("⚠️ Could not find Abundance/Count/Value column in data")
                fig = None
        else:
            # WIDE or WIDE-STRATIFIED format
            if is_stratified:
                # Aggregate columns for selected taxon or PGPT
                st.markdown(f"### Distribution of **{feature_col}** by **{group_col}**")

                # Find all columns that match the selected entity
                if view_by == "Taxon":
                    # Get all columns starting with "taxon|"
                    matching_cols = [col for col in feature_cols if col.startswith(feature_col + '|')]
                else:  # PGPT
                    # Get all columns ending with "|pgpt"
                    matching_cols = [col for col in feature_cols if col.endswith('|' + feature_col)]

                if matching_cols:
                    # Aggregate abundances across matching columns for each sample
                    # CLR data: use MEAN to preserve scale | Raw data: use SUM
                    if is_clr:
                        df_plot = df_merged.with_columns(
                            pl.mean_horizontal([pl.col(c) for c in matching_cols]).alias("Aggregated_Abundance")
                        )
                        agg_method = "MEAN"
                    else:
                        df_plot = df_merged.with_columns(
                            pl.sum_horizontal([pl.col(c) for c in matching_cols]).alias("Aggregated_Abundance")
                        )
                        agg_method = "SUM"

                    st.caption(f"📊 Aggregating {len(matching_cols)} features using {agg_method}: {view_by} = {feature_col}")

                    # Create boxplot with aggregated values
                    fig = px.box(
                        df_plot,
                        x=group_col,
                        y="Aggregated_Abundance",
                        color=group_col,
                        points="all",
                        title=f"Distribution of {feature_col} ({view_by}) by {group_col}"
                    )
                else:
                    st.error(f"⚠️ No features found for {view_by} = {feature_col}")
                    fig = None
            else:
                # Regular wide format
                st.markdown(f"### Distribution of **{feature_col}** by **{group_col}**")

                fig = px.box(
                    df_merged,
                    x=group_col,
                    y=feature_col,
                    color=group_col,
                    points="all",  # Show all points
                    title=f"Distribution of {feature_col} by {group_col}"
                )

        if fig:
            fig.update_layout(
                height=500,
                showlegend=True,
                template="plotly_white"
            )
            # High-resolution download config
            config = {
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': f'boxplot_{feature_col}_{group_col}',
                    'height': 1080,
                    'width': 1920,
                    'scale': 2  # 2x resolution for high quality
                },
                'displayModeBar': True,
                'displaylogo': False
            }
            st.plotly_chart(fig, width='stretch', config=config)

    with viz_tab2:
        st.markdown("### Scatter Plot")

        if format_type_normalized == "long":
            # LONG FORMAT: Cross-Feature Analysis (Taxon vs PGPT)
            # Allows plotting "Bacteroidaceae abundance" vs "NITROGEN_FIXATION abundance"
            # where dots are SAMPLES
            st.info("🔬 **LONG Format Mode**: Compare Taxon abundance vs PGPT abundance across samples")

            # Detect abundance column
            abundance_col = None
            for col in df_merged.columns:
                if col.lower() in ['abundance', 'count', 'value', 'total_pgpt_abundance']:
                    abundance_col = col
                    break

            if not abundance_col:
                st.error("⚠️ Could not find abundance column")
            else:
                # Separate Taxon and PGPT columns
                # Common taxonomy column names: Taxonomy, Taxon, Family, LV3, Genus, Species
                # Common PGPT column names: PGPT, Function, Pathway, Feature
                taxon_col = None
                pgpt_col = None

                for col in feature_cols:
                    col_lower = col.lower()
                    if any(tax in col_lower for tax in ['taxonomy', 'taxon', 'family', 'lv3', 'genus', 'species']):
                        taxon_col = col
                    elif any(func in col_lower for func in ['pgpt', 'function', 'pathway', 'feature']):
                        pgpt_col = col

                # If we found both Taxon and PGPT columns, use separated lists
                if taxon_col and pgpt_col:
                    # Extract unique values for each
                    taxon_values = sorted(df_merged[taxon_col].unique().to_list())
                    pgpt_values = sorted(df_merged[pgpt_col].unique().to_list())

                    if not taxon_values or not pgpt_values:
                        st.error("⚠️ **No entities found**: The feature columns appear to be empty")
                        st.stop()

                    col_x, col_y, col_color = st.columns(3)

                    with col_x:
                        selected_taxon = st.selectbox(
                            "X-axis (Taxon):",
                            options=taxon_values,
                            key="scatter_taxon_long",
                            help="Select a taxonomic group"
                        )

                    with col_y:
                        selected_pgpt = st.selectbox(
                            "Y-axis (PGPT):",
                            options=pgpt_values,
                            key="scatter_pgpt_long",
                            help="Select a functional category"
                        )

                    with col_color:
                        scatter_color = st.selectbox(
                            "Color by:",
                            options=["None"] + metadata_cols,
                            key="scatter_color_long"
                        )

                    # Data transformation: Filter -> GroupBy Sample -> Aggregate Abundances -> Join
                    # CLR data: use MEAN to preserve scale | Raw data: use SUM
                    agg_func = pl.col(abundance_col).mean() if is_clr else pl.col(abundance_col).sum()

                    # Step A: Get abundances for selected taxon across samples
                    df_x = (
                        df_merged
                        .filter(pl.col(taxon_col) == selected_taxon)
                        .group_by("Sample")
                        .agg(agg_func.alias("Taxon_Abundance"))
                    )

                    # Step B: Get abundances for selected PGPT across samples
                    df_y = (
                        df_merged
                        .filter(pl.col(pgpt_col) == selected_pgpt)
                        .group_by("Sample")
                        .agg(agg_func.alias("PGPT_Abundance"))
                    )

                    # Step C: Join X and Y
                    df_scatter = df_x.join(df_y, on="Sample", how="inner")

                    # Count how many rows contributed to each aggregation
                    n_taxon_rows = df_merged.filter(pl.col(taxon_col) == selected_taxon).height
                    n_pgpt_rows = df_merged.filter(pl.col(pgpt_col) == selected_pgpt).height

                    # Step D: Join with metadata for coloring
                    if scatter_color != "None":
                        df_metadata_subset = df_merged.select(["Sample", scatter_color]).unique()
                        df_scatter = df_scatter.join(df_metadata_subset, on="Sample", how="left")

                    st.caption(f"📊 X-axis: aggregating {n_taxon_rows} rows | Y-axis: aggregating {n_pgpt_rows} rows")

                    # Create scatter plot
                    fig_scatter = px.scatter(
                        df_scatter,
                        x="Taxon_Abundance",
                        y="PGPT_Abundance",
                        color=scatter_color if scatter_color != "None" else None,
                        hover_name="Sample",
                        title=f"{selected_pgpt} vs {selected_taxon} (Dots = Samples)",
                        labels={"Taxon_Abundance": selected_taxon, "PGPT_Abundance": selected_pgpt}
                    )

                    st.caption(f"📊 Showing {len(df_scatter)} samples")

                else:
                    # Fallback: Use old generic entity selection if can't separate Taxon/PGPT
                    st.warning(f"⚠️ Could not detect separate Taxon and PGPT columns. Using generic entity selection.")
                    st.info(f"Detected feature columns: {feature_cols}")

                    # Get all unique values from all feature columns combined
                    all_entities = {}
                    for feat_col in feature_cols:
                        unique_vals = df_merged[feat_col].unique().to_list()
                        for val in unique_vals:
                            if val not in all_entities:
                                all_entities[val] = feat_col

                    entity_list = sorted(all_entities.keys())

                    if not entity_list:
                        st.error("⚠️ **No entities found**: The feature columns appear to be empty")
                        st.stop()

                    col_x, col_y, col_color = st.columns(3)

                    with col_x:
                        entity_x = st.selectbox(
                            "X-axis Entity:",
                            options=entity_list,
                            key="scatter_entity_x",
                            help="Select an entity"
                        )

                    with col_y:
                        entity_y = st.selectbox(
                            "Y-axis Entity:",
                            options=entity_list,
                            index=min(1, len(entity_list) - 1),
                            key="scatter_entity_y",
                            help="Select a different entity"
                        )

                    with col_color:
                        scatter_color = st.selectbox(
                            "Color by (Metadata):",
                            options=["None"] + metadata_cols,
                            key="scatter_color_long"
                        )

                    # Data transformation: Filter -> GroupBy Sample -> Aggregate Abundances -> Join
                    # CLR data: use MEAN to preserve scale | Raw data: use SUM
                    agg_func_fallback = pl.col(abundance_col).mean() if is_clr else pl.col(abundance_col).sum()

                    col_x_source = all_entities[entity_x]
                    df_x = (
                        df_merged
                        .filter(pl.col(col_x_source) == entity_x)
                        .group_by("Sample")
                        .agg(agg_func_fallback.alias("Value_X"))
                    )

                    col_y_source = all_entities[entity_y]
                    df_y = (
                        df_merged
                        .filter(pl.col(col_y_source) == entity_y)
                        .group_by("Sample")
                        .agg(agg_func_fallback.alias("Value_Y"))
                    )

                    # Step C: Join X and Y
                    df_scatter = df_x.join(df_y, on="Sample", how="inner")

                    # Step D: Join with metadata for coloring
                    if scatter_color != "None":
                        df_metadata_subset = df_merged.select(["Sample", scatter_color]).unique()
                        df_scatter = df_scatter.join(df_metadata_subset, on="Sample", how="left")

                    # Create scatter plot with samples as dots
                    fig_scatter = px.scatter(
                        df_scatter,
                        x="Value_X",
                        y="Value_Y",
                        color=scatter_color if scatter_color != "None" else None,
                        hover_name="Sample",
                        title=f"{entity_y} vs {entity_x} (Dots = Samples)",
                        labels={"Value_X": entity_x, "Value_Y": entity_y}
                    )

                    st.caption(f"📊 Showing {len(df_scatter)} samples")

        elif is_stratified:
            # WIDE-STRATIFIED: Taxon vs PGPT scatter plot
            st.info("🔬 **WIDE-STRATIFIED Mode**: Compare Taxon abundance vs PGPT abundance across samples")

            col_x, col_y, col_color = st.columns(3)

            with col_x:
                selected_taxon = st.selectbox(
                    "X-axis (Taxon):",
                    options=taxon_list,
                    key="scatter_taxon",
                    help="Select a taxonomic group"
                )

            with col_y:
                selected_pgpt = st.selectbox(
                    "Y-axis (PGPT):",
                    options=pgpt_list,
                    key="scatter_pgpt",
                    help="Select a functional category"
                )

            with col_color:
                scatter_color = st.selectbox(
                    "Color by:",
                    options=["None"] + metadata_cols,
                    key="scatter_color_strat"
                )

            # Aggregate columns for selected taxon and PGPT
            # Find all columns for the taxon (all PGPTs for this taxon)
            taxon_cols = [col for col in feature_cols if col.startswith(selected_taxon + '|')]
            # Find all columns for the PGPT (all taxons with this PGPT)
            pgpt_cols = [col for col in feature_cols if col.endswith('|' + selected_pgpt)]

            if taxon_cols and pgpt_cols:
                # Create aggregated dataframe
                # CLR data: use MEAN to preserve scale | Raw data: use SUM
                if is_clr:
                    df_scatter = df_merged.with_columns([
                        pl.mean_horizontal([pl.col(c) for c in taxon_cols]).alias("Taxon_Abundance"),
                        pl.mean_horizontal([pl.col(c) for c in pgpt_cols]).alias("PGPT_Abundance")
                    ]).select(["Sample", "Taxon_Abundance", "PGPT_Abundance"] + metadata_cols)
                    agg_method = "averaging"
                else:
                    df_scatter = df_merged.with_columns([
                        pl.sum_horizontal([pl.col(c) for c in taxon_cols]).alias("Taxon_Abundance"),
                        pl.sum_horizontal([pl.col(c) for c in pgpt_cols]).alias("PGPT_Abundance")
                    ]).select(["Sample", "Taxon_Abundance", "PGPT_Abundance"] + metadata_cols)
                    agg_method = "summing"

                st.caption(f"📊 X-axis: {agg_method} {len(taxon_cols)} columns | Y-axis: {agg_method} {len(pgpt_cols)} columns")

                # Create scatter plot
                fig_scatter = px.scatter(
                    df_scatter,
                    x="Taxon_Abundance",
                    y="PGPT_Abundance",
                    color=scatter_color if scatter_color != "None" else None,
                    hover_name="Sample",
                    title=f"{selected_pgpt} vs {selected_taxon}",
                    labels={"Taxon_Abundance": selected_taxon, "PGPT_Abundance": selected_pgpt}
                )
            else:
                st.error(f"⚠️ No features found for Taxon={selected_taxon} or PGPT={selected_pgpt}")
                fig_scatter = None

        else:
            # REGULAR WIDE FORMAT: Standard scatter (feature vs feature)
            col_x, col_y, col_color = st.columns(3)

            with col_x:
                scatter_x = st.selectbox(
                    "X-axis:",
                    options=feature_cols,
                    key="scatter_x"
                )

            with col_y:
                scatter_y = st.selectbox(
                    "Y-axis:",
                    options=feature_cols,
                    index=min(1, len(feature_cols) - 1),
                    key="scatter_y"
                )

            with col_color:
                scatter_color = st.selectbox(
                    "Color by:",
                    options=["None"] + metadata_cols,
                    key="scatter_color"
                )

            # Create scatter plot (Plotly supports Polars)
            fig_scatter = px.scatter(
                df_merged,
                x=scatter_x,
                y=scatter_y,
                color=scatter_color if scatter_color != "None" else None,
                hover_name="Sample" if "Sample" in df_merged.columns else None,
                title=f"{scatter_y} vs {scatter_x}"
            )

        # Common styling and display
        if fig_scatter is not None:
            fig_scatter.update_traces(marker=dict(size=10, opacity=0.7))
            fig_scatter.update_layout(
                height=500,
                template="plotly_white"
            )

            # High-resolution download config
            if format_type_normalized == "long":
                filename = f'scatter_{entity_y}_vs_{entity_x}_samples'
            elif is_stratified:
                filename = f'scatter_{selected_pgpt}_vs_{selected_taxon}'
            else:
                filename = f'scatter_{scatter_y}_vs_{scatter_x}'

            config_scatter = {
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': filename,
                    'height': 1080,
                    'width': 1920,
                    'scale': 2
                },
                'displayModeBar': True,
                'displaylogo': False
            }
            st.plotly_chart(fig_scatter, width='stretch', config=config_scatter)

    with viz_tab3:
        st.markdown(f"### Summary Statistics for **{feature_col}** by **{group_col}**")

        # Calculate summary statistics based on format
        if format_type_normalized == "long":
            # For long format, compute stats on Abundance column
            abundance_col = None
            for col in df_merged.columns:
                if col.lower() in ['abundance', 'count', 'value', 'total_pgpt_abundance']:
                    abundance_col = col
                    break

            if abundance_col:
                summary = (
                    df_merged
                    .group_by(group_col)
                    .agg([
                        pl.col(abundance_col).count().alias("N"),
                        pl.col(abundance_col).mean().alias("Mean"),
                        pl.col(abundance_col).std().alias("Std"),
                        pl.col(abundance_col).min().alias("Min"),
                        pl.col(abundance_col).median().alias("Median"),
                        pl.col(abundance_col).max().alias("Max"),
                    ])
                    .with_columns(
                        (pl.col("Std") / pl.col("Mean") * 100).alias("CV%")
                    )
                    .sort(group_col)
                )
            else:
                st.error("⚠️ Could not find Abundance/Count/Value column in data")
                summary = None
        elif is_stratified:
            # For wide-stratified format, use the aggregated abundance column from boxplot
            # Check if aggregated plot was created
            if matching_cols:
                # CLR data: use MEAN to preserve scale | Raw data: use SUM
                if is_clr:
                    df_aggregated = df_merged.with_columns(
                        pl.mean_horizontal([pl.col(c) for c in matching_cols]).alias("Aggregated_Abundance")
                    )
                else:
                    df_aggregated = df_merged.with_columns(
                        pl.sum_horizontal([pl.col(c) for c in matching_cols]).alias("Aggregated_Abundance")
                    )
                summary = (
                    df_aggregated
                    .group_by(group_col)
                    .agg([
                        pl.col("Aggregated_Abundance").count().alias("N"),
                        pl.col("Aggregated_Abundance").mean().alias("Mean"),
                        pl.col("Aggregated_Abundance").std().alias("Std"),
                        pl.col("Aggregated_Abundance").min().alias("Min"),
                        pl.col("Aggregated_Abundance").median().alias("Median"),
                        pl.col("Aggregated_Abundance").max().alias("Max"),
                    ])
                    .with_columns(
                        (pl.col("Std") / pl.col("Mean") * 100).alias("CV%")
                    )
                    .sort(group_col)
                )
            else:
                st.error(f"⚠️ No features found for {view_by} = {feature_col}")
                summary = None
        else:
            # For regular wide format, compute stats on selected feature column
            summary = (
                df_merged
                .group_by(group_col)
                .agg([
                    pl.col(feature_col).count().alias("N"),
                    pl.col(feature_col).mean().alias("Mean"),
                    pl.col(feature_col).std().alias("Std"),
                    pl.col(feature_col).min().alias("Min"),
                    pl.col(feature_col).median().alias("Median"),
                    pl.col(feature_col).max().alias("Max"),
                ])
                .with_columns(
                    (pl.col("Std") / pl.col("Mean") * 100).alias("CV%")
                )
                .sort(group_col)
            )

        # Streamlit supports Polars DataFrames directly
        if summary is not None:
            st.dataframe(
                summary,
                width='stretch',
                height=400
            )

    with viz_tab4:
        st.markdown("### 📊 Statistical Hypothesis Tests")
        st.markdown("Run inferential tests to compare groups (using functions from `stage2_analysis/statistics.py`)")

        # Only available for wide format with numeric features
        if format_type_normalized == "wide":
            # CRITICAL: Detect if group_col is continuous or categorical
            unique_groups = df_merged[group_col].unique().to_list()
            num_groups = len(unique_groups)

            # Check if variable is continuous (>10 unique numeric values)
            is_numeric = df_merged[group_col].dtype.is_numeric()
            is_continuous = is_numeric and num_groups > 10

            if is_continuous:
                st.warning(f"⚠️ **{group_col}** appears to be a CONTINUOUS variable ({num_groups} unique values: {min(unique_groups):.2f} - {max(unique_groups):.2f})")
                st.info("💡 **Recommendation:** Continuous variables (pH, temperature, etc.) should use **Correlation Analysis**, not group comparisons.")

                # Offer correlation analysis instead
                st.markdown("---")
                st.markdown("#### 🔗 Correlation Analysis")
                st.markdown(f"Compute correlations between **{group_col}** and all features")

                corr_method = st.radio(
                    "Correlation method:",
                    ["Spearman (rank-based, robust)", "Pearson (linear)"],
                    help="Spearman is recommended for non-normal distributions (microbiome data)"
                )

                if st.button("🧪 Run Correlation Analysis", type="primary"):
                    with st.spinner("Computing correlations..."):
                        import scipy.stats as ss

                        # Extract continuous variable values
                        metadata_values = df_merged[group_col].to_numpy()

                        # Compute correlation for each feature
                        correlation_results = []
                        for feat in feature_cols:
                            feature_values = df_merged[feat].to_numpy()

                            # Remove NaN pairs
                            mask = ~(np.isnan(metadata_values) | np.isnan(feature_values))
                            if mask.sum() < 3:  # Need at least 3 points
                                continue

                            x = metadata_values[mask]
                            y = feature_values[mask]

                            if "Spearman" in corr_method:
                                corr, pval = ss.spearmanr(x, y)
                            else:
                                corr, pval = ss.pearsonr(x, y)

                            correlation_results.append({
                                "Feature": feat,
                                "correlation": corr,
                                "p_value": pval
                            })

                        # Convert to Polars DataFrame
                        results = pl.DataFrame(correlation_results)

                        # Add FDR correction
                        q_values = fdr_correction(results["p_value"], method='fdr_bh', alpha=0.05)
                        results = results.with_columns(q_values.alias("q_value"))

                        # Add significance markers
                        results = results.with_columns([
                            (pl.col("p_value") < 0.05).alias("p_sig"),
                            (pl.col("q_value") < 0.05).alias("q_sig"),
                            pl.col("correlation").abs().alias("abs_correlation")
                        ])

                        st.success(f"✅ Correlation analysis completed for {results.height} features!")

                        # Summary metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Features", results.height)
                        with col2:
                            n_p_sig = results.filter(pl.col("p_sig")).height
                            st.metric("p < 0.05", f"{n_p_sig} ({100*n_p_sig/results.height:.1f}%)")
                        with col3:
                            n_q_sig = results.filter(pl.col("q_sig")).height
                            st.metric("q < 0.05 (FDR)", f"{n_q_sig} ({100*n_q_sig/results.height:.1f}%)")

                        # Show top correlations
                        results_sorted = results.sort("abs_correlation", descending=True)
                        st.dataframe(results_sorted, width='stretch', height=400)

                        # Download button
                        csv_stats = results_sorted.write_csv()
                        st.download_button(
                            label="📥 Download Correlation Results as CSV",
                            data=csv_stats,
                            file_name=f"correlation_{group_col}_{corr_method.split()[0]}.csv",
                            mime="text/csv"
                        )

                # Don't show group comparison tests for continuous variables
                test_options = []
            else:
                st.info(f"🔢 Detected **{num_groups}** groups in **{group_col}**: {', '.join(map(str, unique_groups[:10]))}{' ...' if num_groups > 10 else ''}")

            # Select test type
            if num_groups >= 3:
                test_options = ["Kruskal-Wallis H-test (3+ groups)", "Mann-Whitney U-test (2 groups)"]
            elif num_groups == 2:
                test_options = ["Mann-Whitney U-test (2 groups)"]
            else:
                st.warning("⚠️ Need at least 2 groups to run statistical tests.")
                test_options = []

            if test_options:
                selected_test = st.radio("Select test:", test_options)

                if st.button("🧪 Run Statistical Test", type="primary"):
                    with st.spinner("Running statistical analysis..."):
                        # Import statistical functions
                        from pgptracker.stage2_analysis.statistics import (
                            kruskal_wallis_test, mann_whitney_u_test, fdr_correction
                        )

                        # Prepare data: need wide N×D format without metadata
                        df_wide_N_D = df_merged.select(["Sample"] + feature_cols)

                        # Metadata with group column
                        df_metadata_for_stats = df_merged.select(["Sample", group_col])

                        if "Kruskal-Wallis" in selected_test:
                            # Run Kruskal-Wallis
                            results = kruskal_wallis_test(
                                df_wide_N_D=df_wide_N_D,
                                metadata=df_metadata_for_stats,
                                sample_id_col="Sample",
                                feature_col="Feature",
                                group_col=group_col,
                                value_col="Abundance"
                            )

                            # Add FDR correction - CRITICAL: pass materialized Series, not Expression
                            q_values = fdr_correction(results["p_value"], method='fdr_bh', alpha=0.05)
                            results = results.with_columns(q_values.alias("q_value"))

                            # Add significance markers
                            results = results.with_columns([
                                (pl.col("p_value") < 0.05).alias("p_sig"),
                                (pl.col("q_value") < 0.05).alias("q_sig")
                            ])

                            st.success(f"✅ Kruskal-Wallis test completed for {results.height} features!")

                        elif "Mann-Whitney" in selected_test:
                            # Select 2 groups to compare
                            col1, col2 = st.columns(2)
                            with col1:
                                group_1 = st.selectbox("Group 1:", unique_groups, key="mw_g1")
                            with col2:
                                group_2 = st.selectbox("Group 2:", unique_groups, index=min(1, len(unique_groups)-1), key="mw_g2")

                            if group_1 != group_2:
                                results = mann_whitney_u_test(
                                    df_wide_N_D=df_wide_N_D,
                                    metadata=df_metadata_for_stats,
                                    sample_id_col="Sample",
                                    feature_col="Feature",
                                    group_col=group_col,
                                    value_col="Abundance",
                                    group_1=group_1,
                                    group_2=group_2
                                )

                                # Add FDR correction - CRITICAL: pass materialized Series, not Expression
                                q_values = fdr_correction(results["p_value"], method='fdr_bh', alpha=0.05)
                                results = results.with_columns(q_values.alias("q_value"))

                                # Add significance markers
                                results = results.with_columns([
                                    (pl.col("p_value") < 0.05).alias("p_sig"),
                                    (pl.col("q_value") < 0.05).alias("q_sig")
                                ])

                                st.success(f"✅ Mann-Whitney U test completed: {group_1} vs {group_2}")
                            else:
                                st.error("⚠️ Please select two different groups.")
                                results = None

                        # Display results
                        if results is not None and results.height > 0:
                            st.markdown("#### 📋 Results:")

                            # Summary metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Features", results.height)
                            with col2:
                                n_p_sig = results.filter(pl.col("p_sig")).height
                                st.metric("p < 0.05", f"{n_p_sig} ({100*n_p_sig/results.height:.1f}%)")
                            with col3:
                                n_q_sig = results.filter(pl.col("q_sig")).height
                                st.metric("q < 0.05 (FDR)", f"{n_q_sig} ({100*n_q_sig/results.height:.1f}%)")

                            # Show significant features first
                            results_sorted = results.sort("p_value")

                            st.dataframe(results_sorted, width='stretch', height=400)

                            # Download button
                            csv_stats = results_sorted.write_csv()
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv_stats,
                                file_name=f"statistical_tests_{group_col}_{selected_test.split()[0]}.csv",
                                mime="text/csv"
                            )
        else:
            st.info("ℹ️ Statistical tests are optimized for WIDE format data. "
                   "Long format statistical testing coming soon!")

    # Side-by-Side Feature Comparison
    st.markdown("---")
    st.markdown("## 🔬 Side-by-Side Feature Comparison")
    st.markdown("Compare multiple features (PGPTs/Taxa) across groups simultaneously")

    with st.expander("⚖️ Compare Features", expanded=False):
        # Let user select multiple features to compare
        if format_type_normalized == "wide":
            # For wide format, select multiple feature columns
            num_features = st.slider(
                "Number of features to compare:",
                min_value=2,
                max_value=min(6, len(feature_cols)),  # Max 6 for readability
                value=2,
                help="Select how many features to display side-by-side"
            )

            selected_features = []
            cols_selector = st.columns(num_features)
            for i, col in enumerate(cols_selector):
                with col:
                    feat = st.selectbox(
                        f"Feature {i+1}:",
                        options=feature_cols,
                        index=min(i, len(feature_cols)-1),
                        key=f"compare_feat_{i}"
                    )
                    selected_features.append(feat)

            # Create side-by-side boxplots
            st.markdown(f"### Comparison by **{group_col}**")

            comparison_cols = st.columns(num_features)
            for i, (col, feat) in enumerate(zip(comparison_cols, selected_features)):
                with col:
                    fig_comp = px.box(
                        df_merged,
                        x=group_col,
                        y=feat,
                        color=group_col,
                        points="all",
                        title=f"{feat[:40]}..." if len(feat) > 40 else feat
                    )
                    fig_comp.update_layout(
                        height=400,
                        showlegend=False,
                        template="plotly_white",
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    config_comp = {
                        'toImageButtonOptions': {
                            'format': 'png',
                            'filename': f'comparison_{feat}_{group_col}',
                            'height': 1080,
                            'width': 720,
                            'scale': 2
                        },
                        'displayModeBar': True,
                        'displaylogo': False
                    }
                    st.plotly_chart(fig_comp, width='stretch', config=config_comp)

                    # Mini summary stats
                    mini_stats = df_merged.group_by(group_col).agg([
                        pl.col(feat).mean().alias("Mean"),
                        pl.col(feat).std().alias("Std")
                    ])
                    st.caption(f"📊 Stats")
                    st.dataframe(mini_stats, height=150, width='content')

        else:
            # For long format: comparison doesn't make as much sense since we plot abundances
            st.info("ℹ️ Side-by-side comparison is optimized for WIDE format. "
                   "In LONG format, use the main boxplot tab to compare different features by switching the selector.")

    # Full data table
    st.markdown("---")
    st.markdown("### 📄 Full Data Table")

    # Initialize with full data
    df_filtered = df_merged

    # Add filters
    with st.expander("🔎 Filter Data"):
        filter_col = st.selectbox("Filter by column:", options=metadata_cols + feature_cols)

        if filter_col in metadata_cols:
            # Categorical filter (Polars native)
            unique_values = df_merged[filter_col].unique().to_list()
            selected_values = st.multiselect(
                f"Select {filter_col} values:",
                options=unique_values,
                default=unique_values
            )
            df_filtered = df_merged.filter(pl.col(filter_col).is_in(selected_values))
        else:
            # Numeric filter (Polars native)
            min_val = float(df_merged[filter_col].min())
            max_val = float(df_merged[filter_col].max())
            range_val = st.slider(
                f"Range for {filter_col}:",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )
            df_filtered = df_merged.filter(
                (pl.col(filter_col) >= range_val[0]) &
                (pl.col(filter_col) <= range_val[1])
            )

        st.info(f"Showing {len(df_filtered)} / {len(df_merged)} samples")

    # Display table
    st.dataframe(
        df_filtered,
        width='stretch',
        height=400
    )

    # Download button (use Polars native CSV export)
    csv = df_filtered.write_csv()
    st.download_button(
        label="💾 Download Filtered Data as CSV",
        data=csv,
        file_name="pgptracker_filtered_data.csv",
        mime="text/csv",
    )
