# src/pgptracker/exports/plot_funcs.py

import matplotlib.pyplot as plt
from matplotlib.figure import Figure as MplFigure
from pathlib import Path
from typing import Union, List, cast, Any
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def export_figure(
    fig: Union[MplFigure, Any],
    base_name: str,
    output_dir: Path,
    formats: List[str] = ['png', 'pdf'],
    dpi: int = 300
) -> None:
    """
    Export single figure to multiple formats.
    Automatically detects matplotlib vs plotly figures.
    
    Args:
        fig: Matplotlib or Plotly figure object.
        base_name: Filename without extension.
        output_dir: Directory to save files.
        formats: List of formats ['png', 'pdf', 'html', 'svg'].
        dpi: Resolution for raster formats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Detect if it's a Plotly figure (checking for write_image method)
    is_plotly = PLOTLY_AVAILABLE and hasattr(fig, 'write_image')
    
    for fmt in formats:
        filepath = output_dir / f"{base_name}.{fmt}"
        
        if is_plotly:
            _export_plotly(cast(Any, fig), filepath, fmt, dpi)
        else:
            _export_matplotlib(cast(MplFigure, fig), filepath, fmt, dpi)
            
        print(f"  -> Saved: {filepath.name}")

def setup_matplotlib_style(style: str = 'seaborn-v0_8-whitegrid') -> None:
    """
    Apply consistent style to all matplotlib plots.
    """
    try:
        plt.style.use(style)
    except OSError:
        # Fallback if style not found
        plt.style.use('ggplot')
    
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3})

def _export_matplotlib(fig: MplFigure, 
                       filepath: Path, 
                       fmt: str, dpi: int) -> None:
    if fmt not in ['png', 'pdf', 'svg']:
        return # Skip unsupported formats silently or log warning
        
    fig.savefig(
        filepath, 
        format=fmt, 
        dpi=dpi, 
        bbox_inches='tight', 
        facecolor='white', 
        edgecolor='none'
    )
    # Close figure to free memory
    plt.close(fig)

def _export_plotly(fig: 'go.Figure', filepath: Path, fmt: str, dpi: int) -> None:
    if fmt == 'html':
        fig.write_html(str(filepath))
    elif fmt in ['png', 'pdf', 'svg']:
        # Requires kaleido installed
        try:
            fig.write_image(str(filepath), format=fmt, width=1200, height=800, scale=dpi/100)
        except ValueError:
            print(f"  -> Warning: Could not export Plotly to {fmt} (kaleido missing?)")
    else:
        pass