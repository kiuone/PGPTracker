# test_integration_stage2.py

import shutil
import polars as pl
import numpy as np
from pathlib import Path
import sys
import os
import argparse 
from typing import cast

# Adiciona o src ao path para conseguir importar os módulos
sys.path.append(str(Path.cwd() / "src"))

from pgptracker.stage2_analysis.pipeline_st2 import run_stage2_pipeline

# --- Configuração do Teste ---
TEST_DIR = Path("temp_integration_test")
INPUT_DIR = TEST_DIR / "inputs"
OUTPUT_DIR = TEST_DIR / "outputs"

def setup_dummy_data():
    """Cria dados falsos compatíveis com o pipeline."""
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    INPUT_DIR.mkdir(parents=True)
    
    # 1. Tabela Wide (D x N) - Simulando saída do Estágio 1
    # Features nas linhas, Amostras nas colunas
    df_wide = pl.DataFrame({
        "FeatureID": ["Feat_A", "Feat_B", "Feat_C", "Feat_D", "Feat_E"],
        "S1": [10, 0, 5, 20, 0],
        "S2": [12, 1, 6, 19, 0],
        "S3": [11, 0, 5, 21, 1],  # Grupo A
        "S4": [0, 20, 1, 5, 10],
        "S5": [1, 22, 0, 4, 12],
        "S6": [0, 19, 2, 6, 11],  # Grupo B
    })
    df_wide.write_csv(INPUT_DIR / "feature_table.tsv", separator="\t")
    
    # 2. Metadados
    df_meta = pl.DataFrame({
        "Sample": ["S1", "S2", "S3", "S4", "S5", "S6"],
        "Treatment": ["Control", "Control", "Control", "Stress", "Stress", "Stress"],
        "pH": [7.0, 7.1, 6.9, 5.5, 5.4, 5.6]
    })
    df_meta.write_csv(INPUT_DIR / "metadata.tsv", separator="\t")
    
    return INPUT_DIR / "feature_table.tsv", INPUT_DIR / "metadata.tsv"

class MockArgs:
    """Simula os argumentos que viriam da CLI (argparse)."""
    def __init__(self, table_path, meta_path, out_dir):
        self.input_table = str(table_path)
        self.metadata = str(meta_path)
        self.output_dir = str(out_dir)
        
        # Parâmetros de Normalização
        self.orientation = "D_N"       # Nossos dados fake são D x N
        self.feature_col_name = "FeatureID"
        
        # Parâmetros de Análise
        self.group_col = "Treatment"   # Para Stats e Plots
        self.target_col = "Treatment"  # Para ML (Classificação)
        self.tsne_perplexity = 2.0     # Baixo porque temos poucas amostras
        
        # Flags de Execução
        self.run_stats = True
        self.run_ml = True
        self.ml_type = "classification"

def run_test():
    print("=== INICIANDO SMOKE TEST DO ESTÁGIO 2 ===")
    
    # 1. Setup
    print("[1/4] Criando dados falsos...")
    table_path, meta_path = setup_dummy_data()
    
    # 2. Configurar Argumentos
    args = MockArgs(table_path, meta_path, OUTPUT_DIR)
    
    # 3. Rodar Pipeline
    print("[2/4] Executando run_stage2_pipeline()... (Isso pode demorar uns segundos)")
    try:
        run_stage2_pipeline(cast(argparse.Namespace, args))
        print("      -> Pipeline rodou sem crashar!")
    except SystemExit as e:
        print(f"      -> FALHA: Pipeline chamou sys.exit({e.code})")
        return
    except Exception as e:
        print(f"      -> FALHA CRÍTICA: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Verificação de Arquivos
    print("[3/4] Verificando arquivos de saída...")
    
    expected_files = [
        "normalization/raw_wide_N_D_data",  # Confere se normalizou
        "normalization/clr_wide_N_D_data",
        "diversity/pca_scores.tsv",         # Confere Ordination
        "diversity/pca_plot.png",           # Confere Visualização
        "diversity/alpha_observed_features.png", # Confere Alpha
        "statistics/differential_abundance_results.tsv", # Confere Stats
        "statistics/volcano_plot.png",
        "statistics/heatmap_significant.png",    # Confere Heatmap
        "machine_learning/random_forest_importance.tsv", # Confere ML
        "machine_learning/rf_importance.png"
    ]
    
    missing = []
    for f in expected_files:
        path = OUTPUT_DIR / f
        if path.exists():
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")
            missing.append(f)
            
    if not missing:
        print("\n[4/4] SUCESSO TOTAL! O pipeline gerou todos os arquivos esperados.")
        print(f"      Verifique visualmente os plots em: {OUTPUT_DIR}")
    else:
        print(f"\n[4/4] FALHA: {len(missing)} arquivos não foram gerados.")

if __name__ == "__main__":
    run_test()