# PGPTracker Stage 2 Data Explorer GUI

Interactive Dash application for exploring CLR-transformed feature tables from the PGPTracker pipeline.

## Overview

This modular GUI allows researchers to:
- Upload and merge Stage 2 output files (CLR data + metadata)
- Explore data in an interactive table with filtering and sorting
- Generate dynamic plots (boxplots and scatter plots)
- Analyze feature distributions across metadata groups

## Architecture

```
gui/
├── __init__.py          # Package initialization
├── app.py               # Entry point (Dash app instance)
├── layout.py            # Main layout (containers, tabs)
├── callbacks.py         # All interactivity logic
├── components.py        # Reusable UI components
├── plots.py             # Plotly graph generation
└── ids.py               # Central component ID store
```

## Installation

Install GUI dependencies:

```bash
pip install dash dash-bootstrap-components dash-ag-grid
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

Run from project root:

```bash
python run_gui.py
```

Access the GUI at: `http://localhost:8050`

### Custom Port

```bash
python run_gui.py --port 8080
```

### Production Mode

```bash
python run_gui.py --no-debug
```

## Input Files

The GUI expects two TSV files from Stage 2:

### 1. CLR Data (`clr_wide_N_D.tsv`)

- Format: N×D (Samples × Features)
- Columns: `Sample`, `Feature1_CLR`, `Feature2_CLR`, ...
- Values: CLR-transformed (float, can be negative)

Example:
```
Sample    FamA|PGPT_1_CLR    Feat_B_CLR
S1        -0.523             1.234
S2        0.891              -0.456
```

### 2. Metadata (`metadata.tsv`)

- Must contain a sample ID column: `Sample` or `SampleID`
- Can contain any grouping columns (e.g., `Treatment`, `Site`)
- Column names are flexible

Example:
```
Sample    Treatment    Site    pH
S1        Control      A       6.5
S2        Treated      B       7.2
```

## Features

### Tab 1: Data Explorer

- **Interactive Table**: View merged data with filtering and sorting
- **Pagination**: Navigate large datasets efficiently
- **Column Filters**: Filter by any column value
- **Sortable**: Click headers to sort

### Tab 2: Interactive Plots

#### Boxplot
- **X-axis**: Metadata grouping column (e.g., Treatment)
- **Y-axis**: Selected feature (e.g., IAA_Synthesis_CLR)
- **Shows**: Distribution of feature values across groups

#### Scatter Plot
- **X-axis**: Any numeric column
- **Y-axis**: Any numeric column
- **Color**: Optional metadata grouping
- **Interactive**: Hover to see sample names

## Code Integration

The GUI is designed to be extended with existing Stage 2 analysis functions:

```python
# Example: Add diversity analysis
from pgptracker.stage2_analysis.diversity import calculate_alpha_diversity

# Use in callbacks to generate additional plots
```

## Development

### Adding New Components

1. Define new IDs in `ids.py`
2. Create reusable components in `components.py`
3. Add to layout in `layout.py`
4. Implement interactivity in `callbacks.py`

### Adding New Plots

1. Create plot function in `plots.py`
2. Add plot controls in `components.py`
3. Register callback in `callbacks.py`

### Testing

Test imports:
```bash
python -c "from pgptracker.gui import app, server, run_app; print('Success')"
```

## Notes

- Uses **Polars** for efficient data handling
- Uses **Dash Bootstrap Components** for responsive UI
- All component IDs centralized in `ids.py` for maintainability
- Callbacks use `@callback` decorator for automatic registration
