"""
Testes unitários para o módulo de feature engineering.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer

import tempfile
import os

from src.feature_engineering import (
    create_derived_features,
    get_numeric_features,
    get_categorical_features,
    create_preprocessing_pipeline,
    fit_preprocessor,
    transform_data,
    get_feature_names,
    save_preprocessor,
    load_preprocessor,
    prepare_features_for_training,
)


@pytest.fixture
def sample_df():
    """DataFrame de exemplo para testes."""
    return pd.DataFrame({
        'inde': [7.5, 8.0, 6.5, 7.0],
        'iaa': [8.0, 7.5, 7.0, 8.5],
        'ieg': [7.0, 8.0, 6.0, 7.5],
        'ips': [6.5, 7.0, 6.0, 6.8],
        'ida': [7.0, 8.5, 6.0, 7.2],
        'ipv': [7.2, 7.8, 6.5, 7.0],
        'mat': [7.0, 8.0, 6.0, 7.5],
        'por': [6.5, 7.5, 5.5, 7.0],
        'ing': [7.5, 8.0, np.nan, 7.0],
        'idade': [12, 13, 11, 14],
        'ano_ingresso': [2022, 2021, 2023, 2020],
        'ano_dados': [2024, 2024, 2024, 2024],
        'genero': ['Masculino', 'Feminino', 'Masculino', 'Feminino'],
        'instituicao': ['Pública', 'Privada', 'Pública', 'Pública'],
        'pedra': ['Topázio', 'Ametista', 'Quartzo', 'Topázio'],
        'target': [0, 0, 1, 0]
    })


class TestCreateDerivedFeatures:
    """Testes para create_derived_features."""

    def test_creates_media_notas(self, sample_df):
        """Verifica se cria média das notas."""
        result = create_derived_features(sample_df)

        assert 'media_notas' in result.columns
        # Para primeira linha: (7.0 + 6.5 + 7.5) / 3 = 7.0
        assert result['media_notas'].iloc[0] == pytest.approx(7.0, rel=0.1)

    def test_creates_anos_no_programa(self, sample_df):
        """Verifica se cria anos no programa."""
        result = create_derived_features(sample_df)

        assert 'anos_no_programa' in result.columns
        # Para primeira linha: 2024 - 2022 = 2
        assert result['anos_no_programa'].iloc[0] == 2

    def test_creates_media_indicadores(self, sample_df):
        """Verifica se cria média dos indicadores."""
        result = create_derived_features(sample_df)

        assert 'media_indicadores' in result.columns
        assert result['media_indicadores'].notna().all()

    def test_creates_ratio_features(self, sample_df):
        """Verifica se cria features de razão."""
        result = create_derived_features(sample_df)

        assert 'ratio_ida_ieg' in result.columns
        assert 'ratio_ipv_ips' in result.columns

    def test_handles_missing_values(self, sample_df):
        """Verifica se lida com valores faltantes."""
        # ing tem NaN na terceira linha
        result = create_derived_features(sample_df)

        # Media_notas deve ser calculada com os valores disponíveis
        assert result['media_notas'].notna().sum() >= 3


class TestGetNumericFeatures:
    """Testes para get_numeric_features."""

    def test_returns_numeric_columns(self, sample_df):
        """Verifica se retorna colunas numéricas."""
        result = get_numeric_features(sample_df)

        assert 'inde' in result
        assert 'idade' in result
        assert 'mat' in result

    def test_excludes_target(self, sample_df):
        """Verifica se exclui coluna target."""
        result = get_numeric_features(sample_df)

        assert 'target' not in result

    def test_excludes_categorical(self, sample_df):
        """Verifica se exclui colunas categóricas."""
        result = get_numeric_features(sample_df)

        assert 'genero' not in result
        assert 'instituicao' not in result


class TestGetCategoricalFeatures:
    """Testes para get_categorical_features."""

    def test_returns_expected_categorical(self, sample_df):
        """Verifica se retorna colunas categóricas esperadas."""
        result = get_categorical_features(sample_df)

        assert 'genero' in result
        assert 'instituicao' in result

    def test_returns_only_existing_columns(self):
        """Verifica se retorna apenas colunas existentes."""
        df = pd.DataFrame({'genero': ['M', 'F']})
        result = get_categorical_features(df)

        assert 'genero' in result
        assert 'instituicao' not in result


class TestCreatePreprocessingPipeline:
    """Testes para create_preprocessing_pipeline."""

    def test_returns_column_transformer(self):
        """Verifica se retorna ColumnTransformer."""
        result = create_preprocessing_pipeline(
            numeric_features=['idade', 'mat'],
            categorical_features=['genero']
        )

        assert isinstance(result, ColumnTransformer)

    def test_has_numeric_transformer(self):
        """Verifica se tem transformador numérico."""
        result = create_preprocessing_pipeline(
            numeric_features=['idade'],
            categorical_features=['genero']
        )

        transformer_names = [name for name, _, _ in result.transformers]
        assert 'num' in transformer_names

    def test_has_categorical_transformer(self):
        """Verifica se tem transformador categórico."""
        result = create_preprocessing_pipeline(
            numeric_features=['idade'],
            categorical_features=['genero']
        )

        transformer_names = [name for name, _, _ in result.transformers]
        assert 'cat' in transformer_names


class TestFitPreprocessor:
    """Testes para fit_preprocessor."""

    def test_returns_fitted_preprocessor(self, sample_df):
        """Verifica se retorna preprocessador treinado."""
        numeric = ['inde', 'idade', 'mat']
        categorical = ['genero', 'instituicao']

        preprocessor, X_transformed = fit_preprocessor(
            sample_df, numeric, categorical
        )

        assert preprocessor is not None
        assert X_transformed is not None

    def test_transforms_data(self, sample_df):
        """Verifica se transforma os dados."""
        numeric = ['inde', 'idade', 'mat']
        categorical = ['genero', 'instituicao']

        _, X_transformed = fit_preprocessor(
            sample_df, numeric, categorical
        )

        assert isinstance(X_transformed, np.ndarray)
        assert X_transformed.shape[0] == len(sample_df)

    def test_handles_missing_values(self, sample_df):
        """Verifica se lida com valores faltantes."""
        numeric = ['inde', 'ing']  # ing tem NaN
        categorical = ['genero']

        _, X_transformed = fit_preprocessor(
            sample_df, numeric, categorical
        )

        # Não deve ter NaN após transformação
        assert not np.isnan(X_transformed).any()


class TestTransformData:
    """Testes para transform_data."""

    def test_transforms_new_data(self, sample_df):
        """Verifica se transforma novos dados."""
        numeric = ['inde', 'idade']
        categorical = ['genero']

        preprocessor, _ = fit_preprocessor(sample_df, numeric, categorical)

        new_data = pd.DataFrame({
            'inde': [7.0],
            'idade': [15],
            'genero': ['Masculino']
        })

        result = transform_data(new_data, preprocessor)

        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1


class TestCreateDerivedFeaturesAnoIngresso:

    def test_anos_no_programa_without_ano_dados(self):
        """Cobre o elif: apenas ano_ingresso sem ano_dados."""
        df = pd.DataFrame({
            'ano_ingresso': [2020, 2022],
            'mat': [7.0, 8.0],
            'por': [6.5, 7.5],
            'ing': [7.5, 8.0],
        })
        result = create_derived_features(df)
        assert 'anos_no_programa' in result.columns
        assert result['anos_no_programa'].iloc[0] == 2024 - 2020
        assert result['anos_no_programa'].iloc[1] == 2024 - 2022


class TestGetFeatureNames:
    """Testes para get_feature_names."""

    def test_returns_feature_names(self, sample_df):
        """Verifica se retorna nomes das features."""
        numeric = ['inde', 'idade']
        categorical = ['genero']

        preprocessor, _ = fit_preprocessor(sample_df, numeric, categorical)
        result = get_feature_names(preprocessor)

        assert isinstance(result, list)
        assert len(result) > 0

    def test_fallback_when_get_feature_names_out_fails(self, sample_df):
        """Cobre o except AttributeError no get_feature_names."""
        from unittest.mock import MagicMock
        mock_preprocessor = MagicMock()
        mock_preprocessor.get_feature_names_out.side_effect = AttributeError
        mock_preprocessor.n_features_in_ = 5

        result = get_feature_names(mock_preprocessor)

        assert isinstance(result, list)
        assert len(result) == 5
        assert result[0] == 'feature_0'


class TestSaveLoadPreprocessor:
    """Testes para save_preprocessor e load_preprocessor."""

    def test_save_and_load_roundtrip(self, sample_df):
        """Verifica se salva e carrega preprocessador corretamente."""
        numeric = ['inde', 'idade']
        categorical = ['genero']
        preprocessor, _ = fit_preprocessor(sample_df, numeric, categorical)

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name

        try:
            save_preprocessor(preprocessor, temp_path)
            loaded = load_preprocessor(temp_path)

            assert loaded is not None
            result = loaded.transform(sample_df[['inde', 'idade', 'genero']])
            assert result.shape[0] == len(sample_df)
        finally:
            os.unlink(temp_path)


class TestPrepareFeaturesForTraining:
    """Testes para prepare_features_for_training."""

    def test_returns_tuple_of_three(self, sample_df):
        """Verifica se retorna df, numericas, categoricas."""
        result = prepare_features_for_training(sample_df)
        assert len(result) == 3

    def test_returns_dataframe_with_derived_features(self, sample_df):
        """Verifica se DataFrame de saída tem features derivadas."""
        df_out, _, _ = prepare_features_for_training(sample_df)
        assert 'media_notas' in df_out.columns

    def test_returns_nonempty_feature_lists(self, sample_df):
        """Verifica se listas de features não estão vazias."""
        _, numeric, categorical = prepare_features_for_training(sample_df)
        assert len(numeric) > 0
        assert len(categorical) > 0
