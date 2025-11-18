"""
PGPTracker Stage 2 Data Explorer - Streamlit Edition

A streamlit-based GUI for exploring CLR-transformed microbiome data.
"""

import streamlit as st
import sys
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="PGPTracker Stage 2 Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import views
from pgptracker.gui.views import upload, exploration
from pgptracker.gui import state


def main():
    """Main application entry point."""

    # Initialize session state
    state.init_session_state()

    # Professional header
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='margin: 0; font-size: 2.5rem;'>🧬 PGPTracker Stage 2 Data Explorer</h1>
        <p style='margin: 0.5rem 0 0 0; font-size: 1.1rem;'>Interactive exploration of CLR-transformed feature tables</p>
    </div>
    """, unsafe_allow_html=True)

    # Check for auto-load directory from command line
    if len(sys.argv) > 1:
        results_dir_str = sys.argv[1]
        results_dir = Path(results_dir_str)
        if results_dir.exists() and results_dir.is_dir():
            st.session_state.results_dir = results_dir
            # Try to auto-load files
            upload.auto_load_from_directory(results_dir)
        else:
            st.warning(f"⚠️ Provided path does not exist or is not a directory: {results_dir_str}")

    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 📊 Navigation")
        page = st.radio(
            "Select View:",
            ["📁 Upload Data", "🔍 Data Exploration"],
            label_visibility="collapsed"
        )

        # Show data summary if loaded
        if st.session_state.get('data_loaded', False):
            st.markdown("---")
            st.markdown("### ✅ Data Loaded")
            st.metric("Samples", st.session_state.n_samples)
            st.metric("Features", st.session_state.n_features)
            st.metric("Metadata Columns", st.session_state.n_metadata_cols)

    # Route to appropriate view
    if page == "📁 Upload Data":
        upload.render()
    elif page == "🔍 Data Exploration":
        exploration.render()


if __name__ == "__main__":
    main()
