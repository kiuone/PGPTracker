#!/usr/bin/env python3
"""
Script de validação para saídas CLR.

Verifica a "regra de ouro" do CLR: a soma dos valores transformados
por amostra deve ser ~0.

Testa os formatos 'long' e 'wide'.
"""

import polars as pl
import sys
from typing import List

# --- Configuração ---
# Caminhos para seus arquivos de saída
CLR_LONG_FILE = "results/test_run_new_unstrat/clr_outputs/stratified_long_clr.tsv"
CLR_WIDE_FILE = "results/test_run_new_unstrat/clr_outputs/stratified_wide_clr.tsv"

# Colunas de features/índice no arquivo WIDE
# (usadas para excluí-las da soma)
WIDE_FEATURE_COLS = ["Family", "Lv3"]

# Tolerância para erros de ponto flutuante.
# O bug original era 0.6. Um ruído de 1e-8 é aceitável.
TOLERANCE = 1e-8
# --- Fim da Configuração ---


def check_long_format(filepath: str) -> bool:
    """Verifica o arquivo de formato longo (agrupando por 'Sample')."""
    
    print("\n--- 1. Verificando Formato LONGO ---")
    print(f"Arquivo: {filepath}")
    
    try:
        df = pl.read_csv(filepath, separator='\t', has_header=True)
        
        # Verificar se as colunas esperadas existem
        if "Sample" not in df.columns or "CLR_Total_PGPT_Abundance" not in df.columns:
            print("  ❌ FALHA: Faltam colunas 'Sample' ou 'CLR_Total_PGPT_Abundance'.")
            return False

        # Agrupar por amostra e somar os valores CLR
        sample_sums = df.group_by("Sample").agg(
            pl.col("CLR_Total_PGPT_Abundance").sum().alias("CLR_Sum")
        )
        
        min_sum = sample_sums["CLR_Sum"].min()
        max_sum = sample_sums["CLR_Sum"].max()

        # CORREÇÃO PYLANCE:
        # Usar isinstance para garantir que min/max são numéricos antes do abs()
        if not isinstance(min_sum, (int, float)) or \
           not isinstance(max_sum, (int, float)):
            print("  ❌ FALHA: Somas (min/max) não são numéricas. Tabela vazia?")
            return False

        print(f"  Soma Mínima (long): {min_sum}")
        print(f"  Soma Máxima (long): {max_sum}")

        if abs(min_sum) < TOLERANCE and abs(max_sum) < TOLERANCE:
            print(f"  ✅ SUCESSO: Somas (long) estão dentro da tolerância ({TOLERANCE}).")
            return True
        else:
            print(f"  ❌ FALHA: Somas (long) excederam a tolerância ({TOLERANCE}).")
            return False

    except Exception as e:
        print(f"  Erro ao processar arquivo long: {e}", file=sys.stderr)
        return False


def check_wide_format(filepath: str, feature_cols: List[str]) -> bool:
    """Verifica o arquivo de formato wide (somando cada coluna de amostra)."""

    print("\n--- 2. Verificando Formato WIDE ---")
    print(f"Arquivo: {filepath}")
    
    try:
        df_wide = pl.read_csv(filepath, separator='\t', has_header=True)
        
        # Identificar colunas de amostra (todas que NÃO são features)
        sample_cols = [c for c in df_wide.columns if c not in feature_cols]
        
        if not sample_cols:
            print("  ❌ FALHA: Não foram encontradas colunas de amostra.")
            print(f"         (Verifique se WIDE_FEATURE_COLS está correto no script)")
            return False

        # Calcular a soma de cada coluna de amostra
        column_sums = df_wide.select(sample_cols).sum()
        
        # "Derreter" (unpivot) para obter min/max facilmente
        df_sums_long = column_sums.unpivot(
            variable_name="Sample", 
            value_name="CLR_Sum"
        )

        min_sum = df_sums_long["CLR_Sum"].min()
        max_sum = df_sums_long["CLR_Sum"].max()

        # CORREÇÃO PYLANCE:
        if not isinstance(min_sum, (int, float)) or \
           not isinstance(max_sum, (int, float)):
            print("  ❌ FALHA: Somas (min/max) não são numéricas. Tabela vazia?")
            return False
            
        print(f"  Soma Mínima (wide): {min_sum}")
        print(f"  Soma Máxima (wide): {max_sum}")

        if abs(min_sum) < TOLERANCE and abs(max_sum) < TOLERANCE:
            print(f"  ✅ SUCESSO: Somas (wide) estão dentro da tolerância ({TOLERANCE}).")
            return True
        else:
            print(f"  ❌ FALHA: Somas (wide) excederam a tolerância ({TOLERANCE}).")
            return False

    except Exception as e:
        print(f"  Erro ao processar arquivo wide: {e}", file=sys.stderr)
        return False


def main():
    """Função principal para executar ambas as verificações."""
    print("=== Verificação de Validade CLR ===")
    
    long_ok = check_long_format(CLR_LONG_FILE)
    wide_ok = check_wide_format(CLR_WIDE_FILE, WIDE_FEATURE_COLS)
    
    print("\n--- Resumo Final ---")
    if long_ok and wide_ok:
        print("✅✅ Ambas as tabelas (long e wide) foram validadas com sucesso!")
        sys.exit(0) # Sucesso
    else:
        print("❌ Uma ou mais tabelas falharam na validação.")
        sys.exit(1) # Falha


if __name__ == "__main__":
    main()