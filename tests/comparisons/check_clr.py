#!/usr/bin/env python3
"""
Script de validação para saídas CLR.

Verifica a "regra de ouro" do CLR: a soma dos valores transformados
por amostra deve ser ~0.

Testa APENAS o formato 'wide', pois é a única saída CLR
da função clr_normalize.py.
"""

import polars as pl
import sys
from typing import List

# --- Configuração ---
# Caminho para o seu arquivo de saída WIDE.
# ATENÇÃO: O nome do arquivo agora depende do 'base_name' usado!
# Ex: Se base_name era 'stratified.tsv', o arquivo será 'clr_wide_stratified.tsv'.
# Ajuste este caminho para o seu arquivo de saída real.
CLR_WIDE_FILE = "results/test_run_new_unstrat/clr_outputs/clr_wide_stratified_output.tsv" 

# Colunas de features/índice no arquivo WIDE
# (usadas para excluí-las da soma)
WIDE_FEATURE_COLS = ["Family", "Lv3"] # Ajuste conforme necessário

# Tolerância para erros de ponto flutuante.
TOLERANCE = 1e-8
# --- Fim da Configuração ---


def check_wide_format(filepath: str, feature_cols: List[str]) -> bool:
    """Verifica o arquivo de formato wide (somando cada coluna de amostra)."""

    print("\n--- Verificando Formato WIDE (Saída CLR) ---")
    print(f"Arquivo: {filepath}")
    
    try:
        df_wide = pl.read_csv(filepath, separator='\t', has_header=True)
        
        # Identificar colunas de amostra (todas que NÃO são features)
        sample_cols = [c for c in df_wide.columns if c not in feature_cols]
        
        if not sample_cols:
            print("  ❌ FALHA: Não foram encontradas colunas de amostra.")
            print(f"       (Verifique se WIDE_FEATURE_COLS está correto no script)")
            print(f"       Colunas encontradas: {df_wide.columns}")
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

        if not isinstance(min_sum, (int, float)) or \
           not isinstance(max_sum, (int, float)):
            print("   FALHA: Somas (min/max) não são numéricas. Tabela vazia?")
            return False
            
        print(f"  Soma Mínima (wide): {min_sum}")
        print(f"  Soma Máxima (wide): {max_sum}")

        if abs(min_sum) < TOLERANCE and abs(max_sum) < TOLERANCE:
            print(f"   SUCESSO: Somas (wide) estão dentro da tolerância ({TOLERANCE}).")
            return True
        else:
            print(f"   FALHA: Somas (wide) excederam a tolerância ({TOLERANCE}).")
            return False

    except Exception as e:
        print(f"  Erro ao processar arquivo wide: {e}", file=sys.stderr)
        return False


def main():
    """Função principal para executar a verificação wide."""
    print("=== Verificação de Validade CLR ===")
    
    wide_ok = check_wide_format(CLR_WIDE_FILE, WIDE_FEATURE_COLS)
    
    print("\n--- Resumo Final ---")
    if wide_ok:
        print(" A tabela CLR (wide) foi validada com sucesso!")
        sys.exit(0) # Sucesso
    else:
        print(" A tabela CLR (wide) falhou na validação.")
        sys.exit(1) # Falha


if __name__ == "__main__":
    main()