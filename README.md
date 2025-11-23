# PGPTracker: A Bioinformatics Pipeline for Functional Prediction and Analysis

PGPTracker is a command-line interface (CLI) tool designed to automate the complete workflow from 16S rRNA sequencing data to in-depth functional and statistical analysis.

It connects Amplicon Sequence Variants (ASVs) to predicted functions (KEGG Orthologs) and maps them to **Plant Growth-Promoting Traits (PGPTs)**.

---

## Core Workflow

The pipeline is split into two main stages and includes a graphical interface for exploration:

* **Stage 1 (`process`):** Handles data processing (QIIME 2, adapted PICRUSt2) to generate unstratified (Function x Sample) and stratified (Taxon x Function x Sample) abundance tables. Utilizes optimized hybrid batching for memory efficiency.
* **Stage 2 (`analysis`):** Takes the tables from Stage 1 and performs normalization (CLR), statistical analysis (Kruskal-Wallis, PERMANOVA), machine learning (Random Forest, Lasso, Boruta), and generates publication-quality visualizations (PCA, Heatmaps, Volcano Plots).
* **Data Explorer (`gui`):** A browser-based interactive dashboard to explore results, visualize patterns, and compare groups dynamically.

---

## Installation

PGPTracker is a pip-installable package that requires **Conda** to manage its bioinformatics dependencies (QIIME 2 and R).

### Step 1: Create and Activate Base Environment
Create and activate a clean Conda environment (Python 3.10+ recommended).

```bash
conda create -n pgptracker python=3.13
conda activate pgptracker
````

### Step 2: Install PGPTracker

Install the package and its core dependencies from PyPI (or local source).

```bash
pip install pgptracker
```

### Step 3: Run Internal Setup (Mandatory)

This command is **mandatory**. It automatically creates and configures the separate `qiime2` Conda environment that PGPTracker needs to run external tools.

```bash
pgptracker setup
```

-----

## Quick Start

### 1\. Interactive Mode (Recommended for Beginners)

The easiest way to run PGPTracker is via the interactive menu, which prompts for all necessary inputs.

```bash
pgptracker -i
```

### 2\. Command Line Mode (For Scripts/Advanced Users)

#### Stage 1: Process Raw Data

Process your raw sequence data (.qza, .fna, or .biom) into PGPT abundance tables.

> **Note:** The default classifier is greengenes.2024.09 for v4 16S, you can add a custom one if you want or need

```bash
pgptracker process \
    --rep-seqs path/to/dna-sequences.fasta \
    --feature-table path/to/feature-table.biom \
    -o my_project_output \
    --classifier-qza /path/to/custom_classifier.qza \
    --stratified \
    --tax-level Genus \
    --max-nsti 2.0 \
    --pgpt-level Lv3 
```

> **Note:** You can also run each step of the stage 1 separately, try pgptracker -h for more information.

#### Stage 2: Statistical Analysis & ML

Analyze the output against your metadata to find significant differences and predictive features.

```bash
pgptracker analysis \
    -i my_project_output/genus_stratified_pgpt.tsv \
    -m path/to/metadata.tsv \
    -o my_project_output/analysis_results \
    --group-col Treatment \
    --target-col pH \
    --ml-type regression
```

#### GUI: Interactive Exploration

Launch the web dashboard to explore your results visually.

```bash
pgptracker gui
```

-----

## Command Reference

### Main Commands

| Command | Description |
| :--- | :--- |
| `pgptracker process` | (Stage 1) Runs the full bioinformatics pipeline (QIIME2, PICRUSt2, PGPTs). |
| `pgptracker analysis` | (Stage 2) Runs statistical tests, ML, and plotting on a Stage 1 output table. |
| `pgptracker setup` | Installs and configures internal Conda environments. **Must be run once after install.** |
| `pgptracker -i` | Runs the tool in a guided, interactive menu-driven mode. |
| `pgptracker clr` | Applies manual CLR normalization to a specific table. |

### `pgptracker process` (Stage 1) Arguments

| Argument | Description |
| :--- | :--- |
| `--rep-seqs` | Path to representative sequences (`.qza` or `.fna`). **Required.** |
| `--feature-table` | Path to feature table (`.qza` or `.biom`). **Required.** |
| `-o, --output` | Output directory to store results. |
| `--stratified` | Flag to generate stratified (Taxon x Function x Sample) output. |
| `--tax-level` | Taxonomic level for stratification (default: `Genus`). |
| `--pgpt-level` | PGPT hierarchical level to use (default: `Lv3`). |
| `--max-nsti` | NSTI threshold for PICRUSt2 filtering (default: `1.7`). |
| `--chunk-size` | Gene families per chunk for memory optimization (default: `1000`). |
| `-t, --threads` | Number of threads to use (default: auto-detect). |
| `--classifier-qza` | Path to a custom QIIME 2 classifier (default: Greengenes 2024.09). |

### `pgptracker analysis` (Stage 2) Arguments

| Argument | Description |
| :--- | :--- |
| `-i, --input-table` | Path to the feature table (output from `process`). **Required.** |
| `-m, --metadata` | Path to the sample metadata file (TSV format). **Required.** |
| `-o, --output-dir` | Directory to save analysis results. |
| `--group-col` | Metadata column for grouping in plots and statistics (e.g., `'Treatment'`). **Required.** |
| `--target-col` | Metadata column to predict in machine learning. Defaults to `--group-col`. |
| `--ml-type` | Type of ML task: `classification` or `regression`. |
| `--input-format` | Format of input table: `wide`, `long`, `stratified`, or `unstratified`. |
| `--orientation` | Orientation of wide tables: `D_N` (Features x Samples) or `N_D` (Samples x Features). |
| `--no-stats` | Flag to skip statistical tests (Kruskal-Wallis/Mann-Whitney). |
| `--no-ml` | Flag to skip machine learning models. |
| `--tsne-perplexity` | Perplexity parameter for t-SNE (default: `30.0`). |
| `--plot-formats` | List of formats to save plots (e.g., `png pdf svg`). |

### `pgptracker clr` Arguments

| Argument | Description |
| :--- | :--- |
| `-i, --input` | Path to input abundance table. |
| `-o, --output` | Output directory. |
| `--format` | Input format: `wide` or `long`. |
| `--sample-col` | Name of sample column (for long format). |
| `--value-col` | Name of abundance column (for long format). |

-----

## Outputs Structure

PGPTracker generates a structured results folder:

| Directory | Content |
| :--- | :--- |
| `normalization/` | `raw_wide_N_D_data` (Counts), `clr_wide_N_D_data` (Normalized). |
| `diversity/` | `alpha_diversity.tsv`, `pca_scores.tsv`, `tsne_scores.tsv`, PERMANOVA results, and plots. |
| `statistics/` | `differential_abundance_results.tsv` (Kruskal/Mann-Whitney with FDR), Volcano Plots. |
| `machine_learning/` | `random_forest_importance.tsv`, `boruta_selection.tsv`, `lasso_coefficients.tsv`, Feature Importance plots. |
| `picrust2_intermediates/` | Raw output from the PICRUSt2 adapted steps (trees, KO predictions). |
| `taxonomy/` | QIIME 2 classification artifacts and exported TSV files. |

-----

## Performance Tuning

PGPTracker includes a custom **Memory Profiler** to help manage resources on large datasets.

You can enable profiling on any command by adding the `--profile` flag:

```bash
# Profiles using the 'production' preset (warns if >5GB RAM)
pgptracker process [...] --profile

# Profiles using 'debug' preset (more verbose)
pgptracker analysis [...] --profile debug
```

A TSV report and a summary table of memory usage per function will be generated after execution.

-----

## Citing

PGPTracker is built upon the work of many others. Please cite the core tools and databases it uses. If you use the **PGPTracker CLI** in your research, please cite:

### PGPTracker Software

  * Mello, Vivian. **PGPTracker: Integration of soil metagenomic data for correlation of microbial markers with plant biochemical indicators.** (2025). UFPR Palotina.

### PGPTracker & PLaBAse
* Atz, S., Rauh, M., Gautam, A., Huson, D.H. **mgPGPT: Metagenomic analysis of plant growth-promoting traits.** (submitted, 2024, preprint)
* Patz, S., Gautam, A., Becker, M., Ruppel, S., Rodríguez-Palenzuela, P., Huson, D.H. **PLaBAse: A comprehensive web resource for analyzing the plant growth-promoting potential of plant-associated bacteria.** (submitted 2021, preprint)

### Core Dependencies
* **QIIME 2:** Bolyen E, Rideout JR, Dillon MR, et al. (2019). Reproducible, interactive, scalable and extensible microbiome data science using QIIME 2. Nature Biotechnology 37: 852–857.
* **PICRUSt2:** Douglas, G.M., Maffei, V.J., Zaneveld, J.R. et al. (2020). PICRUSt2 for prediction of metagenome functions. Nature Biotechnology 38, 685–688.
* **Greengenes2:** McDonald, D., et al. (2024). Greengenes2 unifies microbial data in a single reference tree. Nature Biotechnology.
<!-- end list -->

```
