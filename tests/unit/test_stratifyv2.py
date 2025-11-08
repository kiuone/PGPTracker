"""
Unit tests for stratify analysis functions in pgptracker.analysis.stratify module.

These tests cover loading input files, aggregation logic, and the main stratify analysis function.
Run with: pytest tests/unit/test_stratifyv2.py -v
"""

import pytest
import polars as pl
import polars.testing as pl_testing
from pathlib import Path
import gzip
import io

# Importe as funções do seu módulo instalado
from pgptracker.analysis.stratify import (
    load_ko_predicted, load_seqtab_with_taxonomy, join_and_calculate_batched,
    aggregate_by_tax_level_ko, aggregate_by_tax_level_sample, generate_stratified_analysis
)
# Mock para a dependência importada pelo módulo stratify
from pgptracker.analysis import unstratified

# --- Fixtures: Dados de Teste Reutilizáveis ---

@pytest.fixture
def tax_level():
    """Nível taxonômico de exemplo para os testes."""
    return "Genus"

@pytest.fixture
def mock_seqtab_df(tax_level):
    """DataFrame fictício de seqtab (Tabela de features)."""
    return pl.DataFrame({
        "OTU/ASV_ID": ["ASV1", "ASV2", "ASV3", "ASV4", "ASV5"],
        tax_level: ["GenusA", "GenusA", "GenusB", None, "GenusB"],
        "Sample1": [10, 5, 0, 10, 8],
        "Sample2": [0, 15, 20, 10, 0],
        "Family": ["FamilyX", "FamilyX", "FamilyY", "FamilyX", "FamilyY"], # Coluna extra
    })

# CORREÇÃO 2: Nova fixture para o DataFrame "limpo"
@pytest.fixture
def mock_seqtab_df_cleaned(tax_level):
    """DataFrame fictício de seqtab COMO ele seria retornado por load_seqtab_with_taxonomy."""
    return pl.DataFrame({
        "OTU/ASV_ID": ["ASV1", "ASV2", "ASV3", "ASV4", "ASV5"],
        tax_level: ["GenusA", "GenusA", "GenusB", None, "GenusB"],
        "Sample1": [10, 5, 0, 10, 8],
        "Sample2": [0, 15, 20, 10, 0],
    })

@pytest.fixture
def mock_ko_df():
    """DataFrame fictício de predições de KO."""
    return pl.DataFrame({
        "OTU/ASV_ID": ["ASV1", "ASV2", "ASV3", "ASV5"], # ASV4 está faltando
        "ko:K001": [1, 2, 0, 1],
        "ko:K002": [0, 1, 3, 0],
        "metadata_test": ["a", "b", "c", "d"] # Coluna extra para ser descartada
    })

@pytest.fixture
def mock_pathways_df():
    """DataFrame fictício de mapeamento KO -> PGPT."""
    return pl.DataFrame({
        "KO": ["ko:K001", "ko:K002", "ko:K002", "ko:K003"],
        "Lv3": ["PGPT_A", "PGPT_A", "PGPT_B", "PGPT_C"], # K002 mapeia para A e B
    })

@pytest.fixture
def expected_tax_abun_df(tax_level):
    """Resultado esperado da Etapa 1: aggregate_by_tax_level_sample."""
    df = pl.DataFrame({
        tax_level: ["GenusA", "GenusA", "GenusB", "GenusB"],
        "Sample": ["Sample1", "Sample2", "Sample1", "Sample2"],
        "Total_Tax_Abundance": [15.0, 15.0, 8.0, 20.0],
    })
    # Ordenar para garantir uma comparação estável
    return df.sort([tax_level, "Sample"])

@pytest.fixture
def expected_tax_ko_df(tax_level):
    """Resultado esperado da Etapa 2: aggregate_by_tax_level_ko."""
    # Lembre-se:
    # GenusA (ASV1, ASV2):
    #   K001: [1, 2] -> mean = 1.5
    #   K002: [1] (zero de ASV1 filtrado) -> mean = 1.0
    # GenusB (ASV3, ASV5):
    #   K001: [1] (zero de ASV3 filtrado) -> mean = 1.0
    #   K002: [3] (zero de ASV5 filtrado) -> mean = 3.0
    df = pl.DataFrame({
        tax_level: ["GenusA", "GenusA", "GenusB", "GenusB"],
        "KO": ["ko:K001", "ko:K002", "ko:K001", "ko:K002"],
        "Avg_Copy_Number": [1.5, 1.0, 1.0, 3.0],
    })
    return df.sort([tax_level, "KO"])


# --- Testes para as Funções ---

def test_load_seqtab_with_taxonomy_success(mocker, tmp_path, mock_seqtab_df, tax_level):
    """Testa o carregamento bem-sucedido da tabela de features."""
    mocker.patch.object(Path, 'exists', return_value=True)
    
    # CORREÇÃO 1: Mock de pl.read_csv com side_effect
    # Esta função simula o comportamento de read_csv sendo chamado duas vezes
    def mock_read_csv(*args, **kwargs):
        if 'n_rows' in kwargs and kwargs['n_rows'] == 0:
            # Esta é a chamada de verificação de cabeçalho
            return mock_seqtab_df
        if 'columns' in kwargs:
            # Esta é a chamada principal de carregamento de dados
            # Ela agora respeita o argumento 'columns'
            return mock_seqtab_df.select(kwargs['columns'])
        # Fallback (não deve ser atingido no código normal)
        return mock_seqtab_df

    # Mock pl.read_csv com a função side_effect
    mocker.patch('polars.read_csv', side_effect=mock_read_csv)
    
    mock_path = tmp_path / "seqtab.tsv"
    
    df = load_seqtab_with_taxonomy(mock_path, tax_level)
    
    # Verifica se as colunas desnecessárias (Family) foram descartadas
    expected_cols = ["OTU/ASV_ID", tax_level, "Sample1", "Sample2"]
    assert sorted(df.columns) == sorted(expected_cols)
    assert len(df) == 5

def test_load_seqtab_not_found(mocker, tmp_path):
    """Testa a falha se a tabela de features não for encontrada."""
    mocker.patch.object(Path, 'exists', return_value=False)
    mock_path = tmp_path / "non_existent.tsv"
    
    with pytest.raises(FileNotFoundError):
        load_seqtab_with_taxonomy(mock_path, "Genus")

def test_load_seqtab_invalid_tax_level(mocker, tmp_path, mock_seqtab_df):
    """Testa a falha se o nível taxonômico não existir."""
    mocker.patch.object(Path, 'exists', return_value=True)
    # Apenas o primeiro mock (verificação de cabeçalho) é necessário aqui
    mocker.patch('polars.read_csv', return_value=mock_seqtab_df)
    
    mock_path = tmp_path / "seqtab.tsv"
    
    with pytest.raises(ValueError, match="Taxonomic level 'Species' not found"):
        load_seqtab_with_taxonomy(mock_path, "Species")

def test_load_ko_predicted_success(mocker, tmp_path, mock_ko_df):
    """Testa o carregamento bem-sucedido das predições de KO."""
    mocker.patch.object(Path, 'exists', return_value=True)
    mocker.patch('polars.read_csv', return_value=mock_ko_df)
    
    mock_path = tmp_path / "ko.tsv.gz"
    
    df = load_ko_predicted(mock_path)
    
    # Verifica se a coluna de metadados foi descartada
    assert "metadata_test" not in df.columns
    assert "ko:K001" in df.columns
    assert len(df) == 4

def test_load_ko_predicted_no_ko_cols(mocker, tmp_path):
    """Testa a falha se nenhuma coluna 'ko:' for encontrada."""
    mocker.patch.object(Path, 'exists', return_value=True)
    mock_df_no_ko = pl.DataFrame({"OTU/ASV_ID": ["A1"], "data": [1]})
    mocker.patch('polars.read_csv', return_value=mock_df_no_ko)
    
    mock_path = tmp_path / "ko.tsv.gz"
    
    with pytest.raises(ValueError, match="No KO columns found"):
        load_ko_predicted(mock_path)

def test_aggregate_by_tax_level_sample(mock_seqtab_df, tax_level, expected_tax_abun_df):
    """Testa a Etapa 1: Agregação de abundância por táxon e amostra."""
    
    # CORREÇÃO 2: Filtre o DataFrame de entrada para simular
    # o que a função load_seqtab_with_taxonomy teria feito.
    # Isso evita o erro de tipo misto (string/int) no unpivot.
    cols_to_use = ["OTU/ASV_ID", tax_level, "Sample1", "Sample2"]
    cleaned_seqtab_df = mock_seqtab_df.select(cols_to_use)

    result_df = aggregate_by_tax_level_sample(cleaned_seqtab_df, tax_level)
    
    # Ordenar para garantir uma comparação estável
    result_df = result_df.sort([tax_level, "Sample"])
    
    # CORREÇÃO: Usando 'check_dtypes' com 's'
    pl_testing.assert_frame_equal(result_df, expected_tax_abun_df, check_dtypes=False)

def test_aggregate_by_tax_level_ko(mock_ko_df, mock_seqtab_df, tax_level, expected_tax_ko_df):
    """Testa a Etapa 2: Agregação de KOs por táxon."""
    result_df = aggregate_by_tax_level_ko(mock_ko_df, mock_seqtab_df, tax_level)
    
    # Ordenar para garantir uma comparação estável
    result_df = result_df.sort([tax_level, "KO"])
    
    # CORREÇÃO: Usando 'check_dtypes' com 's'
    pl_testing.assert_frame_equal(result_df, expected_tax_ko_df, check_dtypes=False)

def test_join_and_calculate_batched(
    tmp_path, 
    tax_level, 
    expected_tax_abun_df, 
    expected_tax_ko_df, 
    mock_pathways_df,
):
    """Testa a Etapa 3: Join, cálculo e escrita em lotes."""
    
    # Caminho de saída fictício
    output_file = tmp_path / "stratify_output.tsv.gz"
    
    # Executar a função (que foi corrigida em join_and_calculate_batched_optimized.py)
    join_and_calculate_batched(
        expected_tax_abun_df,
        expected_tax_ko_df,
        mock_pathways_df,
        output_file,
        tax_level,
        'Lv3'
    )
    
    # Verificar se o arquivo foi criado
    assert output_file.exists()
    
    # Ler o arquivo gzipped de volta para verificar seu conteúdo
    result_df = pl.read_csv(output_file, separator='\t')
        
    # Etapa 3: Agregação final (group_by Taxon, PGPT, Sample)
    expected_final_data = {
        tax_level: ["GenusA", "GenusA", "GenusA", "GenusA", 
                    "GenusB", "GenusB", "GenusB", "GenusB"],
        "Lv3": ["PGPT_A", "PGPT_B", "PGPT_A", "PGPT_B",
                    "PGPT_A", "PGPT_B", "PGPT_A", "PGPT_B"],
        "Sample":  ["Sample1", "Sample1", "Sample2", "Sample2",
                    "Sample1", "Sample1", "Sample2", "Sample2"],
        "Total_PGPT_Abundance": [
            22.5 + 15.0, # GenusA, PGPT_A, S1
            15.0,        # GenusA, PGPT_B, S1
            22.5 + 15.0, # GenusA, PGPT_A, S2
            15.0,        # GenusA, PGPT_B, S2
            8.0 + 24.0,  # GenusB, PGPT_A, S1
            24.0,        # GenusB, PGPT_B, S1
            20.0 + 60.0, # GenusB, PGPT_A, S2
            60.0         # GenusB, PGPT_B, S2
        ]
    }
    expected_df = pl.DataFrame(expected_final_data)
    
    # Ordenar ambos os DFs para garantir uma comparação estável
    sort_cols = [tax_level, 'Lv3', 'Sample']
    result_df = result_df.sort(sort_cols)
    expected_df = expected_df.sort(sort_cols)

    # CORREÇÃO: Usando 'check_dtypes' com 's'
    pl_testing.assert_frame_equal(result_df, expected_df, check_dtypes=False)

def test_generate_stratified_analysis_integration(
    mocker, 
    tmp_path, 
    tax_level,
    # CORREÇÃO 2: Use a fixture 'limpa'
    mock_seqtab_df_cleaned, 
    mock_ko_df, 
    mock_pathways_df,
    expected_tax_abun_df,
    expected_tax_ko_df
):
    """Testa a função orquestradora principal, mockando as E/S."""
    
    # --- CORREÇÃO DO PATCH ---
    # Faça o patch dos nomes dentro do módulo 'stratify', que é onde
    # 'generate_stratified_analysis' irá procurá-los.
    
    # CORREÇÃO 2: Retorne o DataFrame "limpo", como a função real faria
    mocker.patch('pgptracker.analysis.stratify.load_seqtab_with_taxonomy', return_value=mock_seqtab_df_cleaned)
    mocker.patch('pgptracker.analysis.stratify.load_ko_predicted', return_value=mock_ko_df)
    
    # CORREÇÃO 4: O patch deve ser no módulo 'stratify'
    mocker.patch('pgptracker.analysis.stratify.load_pathways_db', return_value=mock_pathways_df)
    
    # Mockar a etapa final (join_and_calculate_batched) para espionar sua chamada
    mock_join_calc = mocker.patch('pgptracker.analysis.stratify.join_and_calculate_batched')

    # Caminhos fictícios
    mock_seqtab_path = tmp_path / "seqtab.tsv"
    mock_ko_path = tmp_path / "ko.tsv.gz"
    output_dir = tmp_path / "output"
    expected_output_path = output_dir / "genus_stratified_pgpt.tsv.gz"

    # Executar a função principal
    result_path = generate_stratified_analysis(
        mock_seqtab_path,
        mock_ko_path,
        output_dir,
        tax_level,
        pgpt_level="Lv3", 
    )
    
    # Verificar se o caminho de saída está correto
    assert result_path == expected_output_path
    assert output_dir.exists()

    # Verificar se a função mockada (join_and_calculate_batched) foi chamada
    assert mock_join_calc.call_count == 1
    
    # Verificar os argumentos passados para a função mockada
    call_args = mock_join_calc.call_args[0]
    
    # Verificar os DataFrames agregados
    tax_abun_arg = call_args[0].sort([tax_level, "Sample"])
    tax_ko_arg = call_args[1].sort([tax_level, "KO"])
    
    # CORREÇÃO: Usando 'check_dtypes' com 's'
    pl_testing.assert_frame_equal(tax_abun_arg, expected_tax_abun_df, check_dtypes=False)
    pl_testing.assert_frame_equal(tax_ko_arg, expected_tax_ko_df, check_dtypes=False)
    
    # Verificar outros argumentos
    pl_testing.assert_frame_equal(call_args[2], mock_pathways_df) # pathways
    assert call_args[3] == expected_output_path # output_path
    assert call_args[4] == tax_level # tax_level

def test_generate_stratified_all_null_tax(mocker, tmp_path, tax_level):
    """Testa a falha se a coluna de táxon selecionada contiver apenas nulos."""
    mock_df_all_null = pl.DataFrame({
        "OTU/ASV_ID": ["ASV1", "ASV2"],
        tax_level: [None, None],
        "Sample1": [10, 5],
    })
    
    mocker.patch('pgptracker.analysis.stratify.load_seqtab_with_taxonomy', return_value=mock_df_all_null)
    
    with pytest.raises(ValueError, match="All values in 'Genus' are null"):
        generate_stratified_analysis(
            tmp_path / "seqtab.tsv",
            tmp_path / "ko.tsv.gz",
            tmp_path / "output",
            tax_level,
            pgpt_level="Lv3", 
        )