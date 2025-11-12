import polars as pl
from polars.testing import assert_frame_equal

# Carregue os dois outputs
df_original_pandas = pl.read_csv("results/test_run_fna2/picrust2_intermediates/KO_metagenome_out/pred_metagenome_unstrat.tsv.gz", separator='\t')
df_novo_polars = pl.read_csv("results/test_run_new_unstrat/KO_metagenome_out/pred_metagenome_unstrat.tsv.gz", separator='\t')

# O 'function' pode ser o índice
df_original_pandas = df_original_pandas.sort("function")
df_novo_polars = df_novo_polars.sort("function")

# # --- CORREÇÃO: Garantir a mesma ordem de colunas ---
# # 1. Pegue a lista de colunas do dataframe original (referência)
# col_order = df_original_pandas.columns

# # 2. Reordene o novo dataframe para bater com a ordem do original
# # Isso selecionará "function" e todas as colunas de amostra na ordem esperada.
# df_novo_polars = df_novo_polars.select(col_order)

col_order = df_novo_polars.columns

df_original_pandas = df_original_pandas.select(col_order)

try:
    # atol = tolerância absoluta. 1e-5 (0.00001) é geralmente seguro.
    assert_frame_equal(
        df_original_pandas, 
        df_novo_polars, 
        check_dtypes=False, # Pandas pode salvar como float64, Polars como float32
        abs_tol=1e-5 
    )
    print("SUCESSO: Os outputs são numericamente idênticos no unstrat ko.")

except AssertionError as e:
    print("FALHA: Os outputs são diferentes no unstrat ko.")
    print(e)

# Carregue os dois outputs
df_original_seqtab_norm = pl.read_csv("results/test_run_fna2/picrust2_intermediates/KO_metagenome_out/seqtab_norm.tsv.gz", separator='\t')
df_novo_seqtab_norm = pl.read_csv("results/test_run_new_unstrat/KO_metagenome_out/seqtab_norm.tsv.gz", separator='\t')

# O 'normalized' pode ser o índice
df_original_seqtab_norm = df_original_seqtab_norm.sort("normalized")
df_novo_seqtab_norm = df_novo_seqtab_norm.sort("normalized")

try:
    # atol = tolerância absoluta. 1e-5 (0.00001) é geralmente seguro.
    assert_frame_equal(
        df_original_seqtab_norm.fill_nan(None), 
        df_novo_seqtab_norm.fill_nan(None), 
        check_dtypes=False, # Pandas pode salvar como float64, Polars como float32
        abs_tol=1e-5 
    )
    print("SUCESSO: Os outputs são numericamente idênticos seqtab norm")

except AssertionError as e:
    print("FALHA: Os outputs são diferentes no unstrat ko.")
    print(e)