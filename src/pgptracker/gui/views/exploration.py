"""
Data exploration view with interactive visualizations.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import polars as pl


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

    # Show format info
    if format_type == "long":
        st.info("📊 **LONG/STRATIFIED Format** detected - Using Abundance column for visualization")
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
        if format_type == "long":
            # For long format, let user select taxonomic/functional column
            feature_col = st.selectbox(
                "Stratification Column:",
                options=feature_cols,
                help="Select the taxonomic or functional stratification column (e.g., Taxonomy, PGPT)"
            )
        else:
            # For wide format, select numeric feature column
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
        if format_type == "long":
            st.markdown(f"### Abundance Distribution by **{group_col}**")
            st.caption(f"Stratified by: {feature_col}")

            # For long format, use Abundance column
            # Detect abundance column name
            abundance_col = None
            for col in df_merged.columns:
                if col.lower() in ['abundance', 'count', 'value']:
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
            st.markdown(f"### Distribution of **{feature_col}** by **{group_col}**")

            # Create boxplot with points (Plotly 5.18+ supports Polars directly)
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

        fig_scatter.update_traces(marker=dict(size=10, opacity=0.7))
        fig_scatter.update_layout(
            height=500,
            template="plotly_white"
        )

        # High-resolution download config
        config_scatter = {
            'toImageButtonOptions': {
                'format': 'png',
                'filename': f'scatter_{scatter_y}_vs_{scatter_x}',
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
        if format_type == "long":
            # For long format, compute stats on Abundance column
            abundance_col = None
            for col in df_merged.columns:
                if col.lower() in ['abundance', 'count', 'value']:
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
        else:
            # For wide format, compute stats on selected feature column
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
        if format_type == "wide":
            # Get unique groups
            unique_groups = df_merged[group_col].unique().to_list()
            num_groups = len(unique_groups)

            st.info(f"🔢 Detected **{num_groups}** groups in **{group_col}**: {', '.join(map(str, unique_groups))}")

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

                            # Add FDR correction
                            results = results.with_columns(
                                fdr_correction(pl.col("p_value"), method='fdr_bh', alpha=0.05).alias("q_value")
                            )

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

                                # Add FDR correction
                                results = results.with_columns(
                                    fdr_correction(pl.col("p_value"), method='fdr_bh', alpha=0.05).alias("q_value")
                                )

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
        if format_type == "wide":
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
                    st.plotly_chart(fig_comp, use_container_width=True, config=config_comp)

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
