"""
Testes unitários para o módulo de pré-processamento.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from src.preprocessing import (
    load_data,
    load_all_years,
    standardize_column_names,
    clean_data,
    create_target,
    remove_leaky_features,
    split_features_target,
    get_feature_columns
)


@pytest.fixture
def sample_df():
    """DataFrame de exemplo para testes."""
    return pd.DataFrame({
        'RA': ['RA-001', 'RA-002', 'RA-003'],
        'Nome': ['Aluno 1', 'Aluno 2', 'Aluno 3'],
        'Gênero': ['Masculino', 'Feminino', 'Masculino'],
        'Idade': [12, 13, 11],
        'INDE 2024': [7.5, 8.0, 6.5],
        'IAA': [8.0, 7.5, 7.0],
        'IEG': [7.0, 8.0, 6.0],
        'IPS': [6.5, 7.0, 6.0],
        'IDA': [7.0, 8.5, 6.0],
        'IPV': [7.2, 7.8, 6.5],
        'IAN': [10.0, 10.0, 5.0],
        'Mat': [7.0, 8.0, 6.0],
        'Por': [6.5, 7.5, 5.5],
        'Ing': [7.5, 8.0, 6.0],
        'Defasagem': [0, 0, -1],
        'Fase Ideal': ['Fase 1', 'Fase 2', 'Fase 1'],
        'Instituição de ensino': ['Pública', 'Privada', 'Pública'],
        'Pedra 2024': ['Topázio', 'Ametista', 'Quartzo'],
        'Ano ingresso': [2022, 2021, 2023],
        'Nº Av': [3, 4, 2]
    })


class TestLoadData:

    def test_returns_dataframe(self):
        mock_df = pd.DataFrame({'col': [1, 2, 3]})
        with patch('src.preprocessing.pd.read_excel', return_value=mock_df):
            result = load_data('fake_path.xlsx')
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_uses_correct_sheet(self):
        mock_df = pd.DataFrame({'col': [1]})
        with patch('src.preprocessing.pd.read_excel', return_value=mock_df) as mock_read:
            load_data('fake_path.xlsx', sheet_name='PEDE2023')
            mock_read.assert_called_once_with('fake_path.xlsx', sheet_name='PEDE2023')


class TestLoadAllYears:

    def test_returns_combined_dataframe(self):
        mock_df = pd.DataFrame({'col': [1, 2]})
        mock_xlsx = MagicMock()
        mock_xlsx.sheet_names = ['PEDE2022', 'PEDE2023']
        with patch('src.preprocessing.pd.ExcelFile', return_value=mock_xlsx), \
             patch('src.preprocessing.pd.read_excel', return_value=mock_df):
            result = load_all_years('fake_path.xlsx')
        assert isinstance(result, pd.DataFrame)
        assert 'ano_dados' in result.columns

    def test_combined_has_all_rows(self):
        mock_df = pd.DataFrame({'col': [1, 2]})
        mock_xlsx = MagicMock()
        mock_xlsx.sheet_names = ['PEDE2022', 'PEDE2023']
        with patch('src.preprocessing.pd.ExcelFile', return_value=mock_xlsx), \
             patch('src.preprocessing.pd.read_excel', return_value=mock_df):
            result = load_all_years('fake_path.xlsx')
        assert len(result) == 4  # 2 linhas × 2 anos

    def test_ano_dados_values(self):
        mock_xlsx = MagicMock()
        mock_xlsx.sheet_names = ['PEDE2022', 'PEDE2024']
        with patch('src.preprocessing.pd.ExcelFile', return_value=mock_xlsx), \
             patch('src.preprocessing.pd.read_excel', side_effect=[
                 pd.DataFrame({'col': [1]}),
                 pd.DataFrame({'col': [2]}),
             ]):
            result = load_all_years('fake_path.xlsx')
        assert set(result['ano_dados'].unique()) == {2022, 2024}


class TestStandardizeColumnNames:
    """Testes para standardize_column_names."""

    def test_renames_columns_correctly(self, sample_df):
        """Verifica se renomeia colunas corretamente."""
        result = standardize_column_names(sample_df)

        assert 'genero' in result.columns
        assert 'idade' in result.columns
        assert 'inde' in result.columns
        assert 'defasagem' in result.columns

    def test_preserves_data(self, sample_df):
        """Verifica se preserva os dados após renomear."""
        result = standardize_column_names(sample_df)

        assert len(result) == len(sample_df)
        assert result['genero'].iloc[0] == 'Masculino'

    def test_handles_missing_columns(self):
        """Verifica se lida com colunas ausentes."""
        df = pd.DataFrame({'RA': ['RA-001'], 'Idade': [12]})
        result = standardize_column_names(df)

        assert 'idade' in result.columns
        assert 'ra' in result.columns


    def test_handles_alternative_defasagem_column(self):
        """Cobre o elif 'Defas' → 'defasagem'."""
        df = pd.DataFrame({'Defas': [0, -1], 'Idade': [12, 13]})
        result = standardize_column_names(df)
        assert 'defasagem' in result.columns

    def test_handles_idade_22_column(self):
        """Cobre o elif 'Idade 22' → 'idade'."""
        df = pd.DataFrame({'Idade 22': [12, 13]})
        result = standardize_column_names(df)
        assert 'idade' in result.columns

    def test_handles_nome_anonimizado_column(self):
        """Cobre o elif 'Nome Anonimizado' → 'nome'."""
        df = pd.DataFrame({'Nome Anonimizado': ['Aluno 1']})
        result = standardize_column_names(df)
        assert 'nome' in result.columns

    def test_handles_fase_ideal_lowercase(self):
        """Cobre o elif 'Fase ideal' → 'fase_ideal'."""
        df = pd.DataFrame({'Fase ideal': ['Fase 1']})
        result = standardize_column_names(df)
        assert 'fase_ideal' in result.columns

    def test_handles_inde_older_versions(self):
        """Cobre elif para INDE 2023, INDE 23, INDE 22."""
        df = pd.DataFrame({'INDE 2023': [7.5]})
        result = standardize_column_names(df)
        assert 'inde' in result.columns

    def test_handles_pedra_older_versions(self):
        """Cobre elif para Pedra 2023, Pedra 23, Pedra 22."""
        df = pd.DataFrame({'Pedra 2023': ['Topázio']})
        result = standardize_column_names(df)
        assert 'pedra' in result.columns


class TestCleanData:
    """Testes para clean_data."""

    def test_removes_duplicate_columns(self):
        """Verifica se remove colunas duplicadas (.1)."""
        df = pd.DataFrame({
            'coluna': [1, 2],
            'coluna.1': [3, 4]
        })
        result = clean_data(df)

        assert 'coluna' in result.columns
        assert 'coluna.1' not in result.columns

    def test_converts_numeric_columns(self, sample_df):
        """Verifica se converte colunas numéricas."""
        df = standardize_column_names(sample_df)
        result = clean_data(df)

        assert result['idade'].dtype in [np.int64, np.float64]
        assert result['inde'].dtype == np.float64


class TestCreateTarget:
    """Testes para create_target."""

    def test_creates_binary_target(self, sample_df):
        """Verifica se cria target binário corretamente."""
        df = standardize_column_names(sample_df)
        result = create_target(df, binary=True)

        assert 'target' in result.columns
        assert set(result['target'].unique()).issubset({0, 1})

    def test_binary_target_values(self, sample_df):
        """Verifica se valores do target binário estão corretos."""
        df = standardize_column_names(sample_df)
        result = create_target(df, binary=True)

        # defasagem = [0, 0, -1] -> target = [0, 0, 1]
        expected = [0, 0, 1]
        assert list(result['target']) == expected

    def test_multiclass_target(self, sample_df):
        """Verifica se cria target multi-classe."""
        df = standardize_column_names(sample_df)
        result = create_target(df, binary=False)

        # Deve manter valores originais da defasagem
        assert list(result['target']) == [0, 0, -1]

    def test_raises_error_without_defasagem(self):
        """Verifica se levanta erro sem coluna defasagem."""
        df = pd.DataFrame({'idade': [12, 13]})

        with pytest.raises(ValueError, match="defasagem"):
            create_target(df)


class TestRemoveLeakyFeatures:
    """Testes para remove_leaky_features."""

    def test_removes_ian(self, sample_df):
        """Verifica se remove IAN (vazamento de dados)."""
        df = standardize_column_names(sample_df)
        result = remove_leaky_features(df)

        assert 'ian' not in result.columns

    def test_removes_fase_ideal(self, sample_df):
        """Verifica se remove fase_ideal."""
        df = standardize_column_names(sample_df)
        result = remove_leaky_features(df)

        assert 'fase_ideal' not in result.columns

    def test_preserves_other_columns(self, sample_df):
        """Verifica se preserva outras colunas."""
        df = standardize_column_names(sample_df)
        result = remove_leaky_features(df)

        assert 'iaa' in result.columns
        assert 'ieg' in result.columns


class TestGetFeatureColumns:
    """Testes para get_feature_columns."""

    def test_excludes_identifiers(self, sample_df):
        """Verifica se exclui identificadores."""
        df = standardize_column_names(sample_df)
        df = create_target(df)
        features = get_feature_columns(df)

        assert 'ra' not in features
        assert 'target' not in features

    def test_includes_valid_features(self, sample_df):
        """Verifica se inclui features válidas."""
        df = standardize_column_names(sample_df)
        df = create_target(df)
        features = get_feature_columns(df)

        # Pelo menos algumas features devem estar presentes
        assert len(features) > 0


class TestSplitFeaturesTarget:
    """Testes para split_features_target."""

    def test_splits_correctly(self, sample_df):
        """Verifica se separa X e y corretamente."""
        df = standardize_column_names(sample_df)
        df = create_target(df)

        X, y = split_features_target(df)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)

    def test_target_not_in_features(self, sample_df):
        """Verifica se target não está nas features."""
        df = standardize_column_names(sample_df)
        df = create_target(df)

        X, y = split_features_target(df)

        assert 'target' not in X.columns

    def test_raises_error_without_target(self, sample_df):
        """Verifica se levanta erro sem coluna target."""
        df = standardize_column_names(sample_df)

        with pytest.raises(ValueError, match="target"):
            split_features_target(df)
