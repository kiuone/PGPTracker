# PGPTracker: A Bioinformatics Pipeline for Functional Prediction and Analysis

PGPTracker is a command-line interface (CLI) tool designed to automate the complete workflow from 16S rRNA sequencing data to in-depth functional and statistical analysis.

It connects Amplicon Sequence Variants (ASVs) to predicted functions (KEGG Orthologs) and maps them to **Plant Growth-Promoting Traits (PGPTs)**.

---

## Core Workflow

The pipeline is split into two main stages:

* **Stage 1 (`process`):** Handles data processing (QIIME 2, PICRUSt2) to generate unstratified (Function x Sample) and stratified (Taxon x Function x Sample) abundance tables.
* **Stage 2 (`analysis`):** Takes the tables from Stage 1 and performs normalization (CLR), statistical analysis (Kruskal-Wallis, PERMANOVA), machine learning (Random Forest, Lasso), and generates publication-quality visualizations (PCA, Heatmaps, Volcano Plots).

---

## Installation

PGPTracker is a pip-installable package that requires **Conda** to manage its bioinformatics dependencies (QIIME 2 and PICRUSt2).

### Step 1: Create and Activate Base Environment
Create and activate a clean Conda environment (Python 3.10+ recommended).

```bash
conda create -n pgptracker python=3.13
conda activate pgptracker
```

### Step 2: Install PGPTracker
Install the package and its core dependencies from PyPI.

```bash
pip install pgptracker
```

### Step 3: Run Internal Setup (Mandatory)
This command is **mandatory**. It automatically creates and configures the separate `qiime2` and `picrust2` Conda environments that PGPTracker needs to run external tools.

```bash
pgptracker setup
```

---

## Quick Start: A Full Example

This example demonstrates the full process and subsequent analysis.

> **Note:** You can also run the command `pgptracker -i` to enter the interactive mode, which is much more user-friendly.

### Step 1: Run Stage 1 (`process`)

Process your raw sequence data (.qza, .fna, or .biom) into PGPT abundance tables. This example generates a table stratified by Genus.

```bash
pgptracker process \
    --rep-seqs path/to/dna-sequences.fasta \
    --feature-table path/to/feature-table.biom \
    -o my_project_output \
    --stratified \
    --tax-level Genus
```
*This command will create the file `my_project_output/genus_stratified_pgpt.tsv`.*

### Step 2: Run Stage 2 (`analysis`)

Analyze the stratified output against your metadata to find which Genus/Function pairs differ by `Treatment`.

```bash
pgptracker analysis \
    -i my_project_output/genus_stratified_pgpt.tsv \
    -m path/to/my_metadata.tsv \
    -o my_project_output/analysis_by_treatment \
    --input-format long \
    --group-col Treatment \
    --target-col Treatment \
    --ml-type classification
```
*This will create the `analysis_by_treatment` directory containing plots and machine learning results.*

---

## Command Reference

### Main Commands

| Command | Description |
| :--- | :--- |
| `pgptracker process` | (Stage 1) Runs the full bioinformatics pipeline (QIIME2, PICRUSt2, PGPTs). |
| `pgptracker analysis` | (Stage 2) Runs statistical tests, ML, and plotting on a Stage 1 output table. |
| `pgptracker setup` | Installs and configures internal Conda environments. **Must be run once after install.** |
| `pgptracker -i` | Runs the tool in a guided, interactive menu-driven mode. |

### `pgptracker process` (Stage 1) Arguments

| Argument | Description |
| :--- | :--- |
| `--rep-seqs` | Path to representative sequences (`.qza` or `.fna`). |
| `--feature-table` | Path to feature table (`.qza` or `.biom`). |
| `-o, --output` | Output directory to store results. |
| `--stratified` | Flag to generate stratified (Taxon x Function x Sample) output. |
| `--tax-level` | Taxonomic level for stratification (default: `Genus`). |
| `--max-nsti` | NSTI threshold for PICRUSt2 filtering (default: `1.7`). |
| `-t, --threads` | Number of threads to use (default: auto-detect). |
| `--classifier-qza` | Path to a custom QIIME 2 classifier (default: Greengenes 2024.09). |

### `pgptracker analysis` (Stage 2) Arguments

| Argument | Description |
| :--- | :--- |
| `-i, --input-table` | Path to the input table (output from `process`). |
| `-m, --metadata` | Path to the sample metadata file (TSV format). |
| `-o, --output-dir` | Directory to save analysis results. |
| `--group-col` | Metadata column for grouping in plots and statistics (e.g., `'Treatment'`). |
| `--target-col` | Metadata column to predict in machine learning (e.g., `'pH'` or `'Treatment'`). |
| `--ml-type` | Type of ML task: `classification` or `regression`. |
| `--input-format` | Format of the input table: `wide` (unstratified) or `long` (stratified). |
| `--no-stats` | Flag to skip statistical tests (Kruskal-Wallis/Mann-Whitney). |
| `--no-ml` | Flag to skip machine learning models. |

---

## Example Workflows (Stage 2 Analysis Cookbook)

### A. Classification: Predict Environmental Biome

**Question:** "Can the functional profile distinguish between biomes (e.g., forest vs. desert)?"

```bash
pgptracker analysis \
    -i path/to/unstratified_pgpt_Lv3_abundances.tsv \
    -m path/to/emp_metadata.tsv \
    -o results/analysis_biome \
    --feature-col-name Lv3 \
    --group-col env_biome \
    --target-col env_biome \
    --ml-type classification
```

### B. Regression: Correlate with Chemistry (pH)

**Question:** "Which bacterial functions (PGPTs) are most associated with soil pH?"

```bash
pgptracker analysis \
    -i path/to/unstratified_pgpt_Lv3_abundances.tsv \
    -m path/to/emp_metadata.tsv \
    -o results/analysis_ph \
    --feature-col-name Lv3 \
    --group-col env_feature \
    --target-col ph \
    --ml-type regression
```

---

## Outputs

PGPTracker generates publication-ready outputs in your results folder:

| Directory | Content |
| :--- | :--- |
| `normalization/` | Raw and CLR-normalized abundance tables. |
| `diversity/` | Alpha Diversity plots (Shannon, Simpson), Beta Diversity plots (PCA, t-SNE), and PERMANOVA results. |
| `statistics/` | Differential Abundance results (Kruskal-Wallis), Volcano Plots, and Clustered Heatmaps. |
| `machine_learning/` | Feature Importance bar plots (Random Forest / Lasso) and Boruta selection results. |

---

## Citing

PGPTracker is built upon the work of many others. Please cite the core tools and databases it uses:

### PGPTracker & PLaBAse
* Atz, S., Rauh, M., Gautam, A., Huson, D.H. **mgPGPT: Metagenomic analysis of plant growth-promoting traits.** (submitted, 2024, preprint)
* Patz, S., Gautam, A., Becker, M., Ruppel, S., Rodríguez-Palenzuela, P., Huson, D.H. **PLaBAse: A comprehensive web resource for analyzing the plant growth-promoting potential of plant-associated bacteria.** (submitted 2021, preprint)

### Core Dependencies
* **QIIME 2:** Bolyen E, Rideout JR, Dillon MR, et al. (2019). Reproducible, interactive, scalable and extensible microbiome data science using QIIME 2. Nature Biotechnology 37: 852–857.
* **PICRUSt2:** Douglas, G.M., Maffei, V.J., Zaneveld, J.R. et al. (2020). PICRUSt2 for prediction of metagenome functions. Nature Biotechnology 38, 685–688.
* **Greengenes2:** McDonald, D., et al. (2024). Greengenes2 unifies microbial data in a single reference tree. Nature Biotechnology.
