"""
Upload/Data Loading view with smart auto-discovery.
"""

import streamlit as st
import polars as pl
from pathlib import Path
from pgptracker.gui import utils


def auto_load_from_directory(results_dir: Path):
    """
    Automatically load CLR data and metadata from results directory.
    Stores data in session state without displaying messages.

    Args:
        results_dir: Path to pipeline results directory
    """
    # Only load once
    if st.session_state.get('auto_load_attempted', False):
        return

    st.session_state.auto_load_attempted = True

    # Look for expected files
    clr_file = results_dir / "clr_wide_N_D.tsv"
    metadata_file = results_dir / "metadata.tsv"

    if clr_file.exists() and metadata_file.exists():
        try:
            # Load files with caching
            df_clr = utils.load_tsv_file(clr_file)
            df_metadata = utils.load_tsv_file(metadata_file)

            # Auto-detect sample column
            metadata_sample_col = utils.auto_detect_sample_column(df_metadata.columns)
            if not metadata_sample_col:
                # Store for manual selection
                st.session_state.pending_metadata = df_metadata
                st.session_state.pending_clr = df_clr
                st.session_state.auto_load_status = "needs_manual_selection"
                return

            # Merge data
            df_merged, metadata_cols, feature_cols = utils.merge_data(
                df_clr, df_metadata, metadata_sample_col
            )

            # Store in session state
            st.session_state.df_merged = df_merged
            st.session_state.metadata_cols = metadata_cols
            st.session_state.feature_cols = feature_cols
            st.session_state.n_samples = df_merged.shape[0]
            st.session_state.n_features = len(feature_cols)
            st.session_state.n_metadata_cols = len(metadata_cols)
            st.session_state.data_loaded = True
            st.session_state.auto_load_status = "success"

        except Exception as e:
            st.session_state.auto_load_status = f"error: {e}"
    else:
        st.session_state.auto_load_status = "files_not_found"


def render():
    """Render the upload/data loading view."""

    st.markdown("## 📁 Data Loading")

    # Show auto-load status
    if st.session_state.get('results_dir'):
        st.info(f"📂 Results directory: `{st.session_state.results_dir}`")

        # Display auto-load status messages
        auto_status = st.session_state.get('auto_load_status')
        if auto_status == "success":
            st.success(f"✅ Auto-loaded data from results directory")
        elif auto_status == "files_not_found":
            st.warning("⚠️ Expected files (clr_wide_N_D.tsv, metadata.tsv) not found in results directory")
        elif auto_status and auto_status.startswith("error:"):
            st.error(f"❌ {auto_status}")
        elif auto_status == "needs_manual_selection":
            st.info("ℹ️ Sample ID column could not be auto-detected. Please select manually below.")

    # Check if data already loaded
    if st.session_state.get('data_loaded', False):
        st.success("✅ Data successfully loaded!")

        # Show summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Samples", st.session_state.n_samples)
        with col2:
            st.metric("Features", st.session_state.n_features)
        with col3:
            st.metric("Metadata Columns", st.session_state.n_metadata_cols)

        # Show preview (Streamlit supports Polars DataFrames directly)
        st.markdown("### 👀 Data Preview")
        st.dataframe(
            st.session_state.df_merged.head(10),
            use_container_width=True,
            height=300
        )

        # Option to reset
        if st.button("🔄 Load Different Data"):
            st.session_state.data_loaded = False
            st.session_state.df_merged = None
            st.rerun()

    else:
        # Manual upload interface
        st.markdown("### 📤 Manual Upload")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 1. Upload Metadata")
            metadata_file = st.file_uploader(
                "metadata.tsv",
                type=["tsv", "txt"],
                key="metadata_upload",
                help="Upload your metadata file (TSV format)"
            )

        with col2:
            st.markdown("#### 2. Upload CLR Data")
            clr_file = st.file_uploader(
                "clr_wide_N_D.tsv",
                type=["tsv", "txt"],
                key="clr_upload",
                help="Upload your CLR-transformed data (TSV format)"
            )

        # Process uploads
        if metadata_file and clr_file:
            with st.spinner("Loading data..."):
                try:
                    # Load files
                    df_metadata = utils.load_uploaded_file(metadata_file)
                    df_clr = utils.load_uploaded_file(clr_file)

                    # Let user select sample column
                    st.markdown("#### 3. Select Sample ID Column")
                    metadata_sample_col = st.selectbox(
                        "Which column contains the sample IDs in metadata?",
                        options=df_metadata.columns,
                        index=0 if not utils.auto_detect_sample_column(df_metadata.columns)
                              else df_metadata.columns.index(utils.auto_detect_sample_column(df_metadata.columns))
                    )

                    if st.button("✅ Load Data", type="primary"):
                        # Merge data
                        df_merged, metadata_cols, feature_cols = utils.merge_data(
                            df_clr, df_metadata, metadata_sample_col
                        )

                        # Store in session state
                        st.session_state.df_merged = df_merged
                        st.session_state.metadata_cols = metadata_cols
                        st.session_state.feature_cols = feature_cols
                        st.session_state.n_samples = df_merged.shape[0]
                        st.session_state.n_features = len(feature_cols)
                        st.session_state.n_metadata_cols = len(metadata_cols)
                        st.session_state.data_loaded = True

                        st.success("✅ Data loaded successfully!")
                        st.rerun()

                except Exception as e:
                    st.error(f"Error loading data: {e}")
