"""
Testes unitários para o pipeline de treinamento.
"""
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.pipeline import run_pipeline


@pytest.fixture
def sample_df():
    """DataFrame com estrutura semelhante ao dado real."""
    np.random.seed(42)
    n = 120
    return pd.DataFrame({
        'inde': np.random.uniform(4, 10, n),
        'iaa': np.random.uniform(4, 10, n),
        'ieg': np.random.uniform(4, 10, n),
        'ips': np.random.uniform(4, 10, n),
        'ipp': np.random.uniform(4, 10, n),
        'ida': np.random.uniform(4, 10, n),
        'ipv': np.random.uniform(4, 10, n),
        'mat': np.random.uniform(4, 10, n),
        'por': np.random.uniform(4, 10, n),
        'ing': np.random.uniform(4, 10, n),
        'idade': np.random.randint(10, 18, n),
        'ano_ingresso': np.random.randint(2018, 2024, n),
        'genero': np.random.choice(['Masculino', 'Feminino'], n),
        'instituicao': np.random.choice(['Pública', 'Particular'], n),
        'pedra': np.random.choice(['Quartzo', 'Ágata', 'Topázio', 'Ametista'], n),
        'defasagem': np.random.randint(-2, 2, n),
    })


def _pipeline_patches(tmp_path):
    """Retorna context manager com todos os patches necessários."""
    return (
        patch('src.pipeline.get_data_path', side_effect=lambda f='', s='': tmp_path / f),
        patch('src.pipeline.get_model_path', side_effect=lambda f='': tmp_path / f),
        patch('src.pipeline.MetricsLogger'),
        patch('src.pipeline.setup_logging', return_value=MagicMock()),
    )


class TestRunPipeline:

    @patch('src.pipeline.save_evaluation_report')
    @patch('src.pipeline.generate_model_card', return_value={})
    @patch('src.pipeline.save_model')
    @patch('src.pipeline.save_preprocessor')
    @patch('src.pipeline.load_data')
    def test_returns_model_and_metrics(
        self, mock_load, mock_save_prep, mock_save_model,
        mock_model_card, mock_save_report, sample_df, tmp_path
    ):
        mock_load.return_value = sample_df
        with patch('src.pipeline.get_data_path', side_effect=lambda f='', s='': tmp_path / f), \
             patch('src.pipeline.get_model_path', side_effect=lambda f='': tmp_path / f), \
             patch('src.pipeline.MetricsLogger'), \
             patch('src.pipeline.setup_logging', return_value=MagicMock()):
            model, metrics = run_pipeline(data_path='dummy.xlsx', model_type='random_forest', compare=False)

        assert model is not None
        assert 'accuracy' in metrics
        assert 'f1' in metrics

    @patch('src.pipeline.save_evaluation_report')
    @patch('src.pipeline.generate_model_card', return_value={})
    @patch('src.pipeline.save_model')
    @patch('src.pipeline.save_preprocessor')
    @patch('src.pipeline.load_data')
    def test_metrics_in_valid_range(
        self, mock_load, mock_save_prep, mock_save_model,
        mock_model_card, mock_save_report, sample_df, tmp_path
    ):
        mock_load.return_value = sample_df
        with patch('src.pipeline.get_data_path', side_effect=lambda f='', s='': tmp_path / f), \
             patch('src.pipeline.get_model_path', side_effect=lambda f='': tmp_path / f), \
             patch('src.pipeline.MetricsLogger'), \
             patch('src.pipeline.setup_logging', return_value=MagicMock()):
            _, metrics = run_pipeline(data_path='dummy.xlsx', compare=False)

        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['f1'] <= 1

    @patch('src.pipeline.save_evaluation_report')
    @patch('src.pipeline.generate_model_card', return_value={})
    @patch('src.pipeline.save_model')
    @patch('src.pipeline.save_preprocessor')
    @patch('src.pipeline.load_data')
    def test_pipeline_with_compare(
        self, mock_load, mock_save_prep, mock_save_model,
        mock_model_card, mock_save_report, sample_df, tmp_path
    ):
        mock_load.return_value = sample_df
        with patch('src.pipeline.get_data_path', side_effect=lambda f='', s='': tmp_path / f), \
             patch('src.pipeline.get_model_path', side_effect=lambda f='': tmp_path / f), \
             patch('src.pipeline.MetricsLogger'), \
             patch('src.pipeline.setup_logging', return_value=MagicMock()):
            model, metrics = run_pipeline(data_path='dummy.xlsx', compare=True)

        assert model is not None
        assert 'accuracy' in metrics

    @patch('src.pipeline.save_evaluation_report')
    @patch('src.pipeline.generate_model_card', return_value={})
    @patch('src.pipeline.save_model')
    @patch('src.pipeline.save_preprocessor')
    @patch('src.pipeline.load_data')
    def test_save_model_and_preprocessor_called(
        self, mock_load, mock_save_prep, mock_save_model,
        mock_model_card, mock_save_report, sample_df, tmp_path
    ):
        mock_load.return_value = sample_df
        with patch('src.pipeline.get_data_path', side_effect=lambda f='', s='': tmp_path / f), \
             patch('src.pipeline.get_model_path', side_effect=lambda f='': tmp_path / f), \
             patch('src.pipeline.MetricsLogger'), \
             patch('src.pipeline.setup_logging', return_value=MagicMock()):
            run_pipeline(data_path='dummy.xlsx', compare=False)

        mock_save_model.assert_called_once()
        mock_save_prep.assert_called_once()

    @patch('src.pipeline.save_evaluation_report')
    @patch('src.pipeline.generate_model_card', return_value={})
    @patch('src.pipeline.save_model')
    @patch('src.pipeline.save_preprocessor')
    @patch('src.pipeline.load_data')
    def test_pipeline_gradient_boosting(
        self, mock_load, mock_save_prep, mock_save_model,
        mock_model_card, mock_save_report, sample_df, tmp_path
    ):
        mock_load.return_value = sample_df
        with patch('src.pipeline.get_data_path', side_effect=lambda f='', s='': tmp_path / f), \
             patch('src.pipeline.get_model_path', side_effect=lambda f='': tmp_path / f), \
             patch('src.pipeline.MetricsLogger'), \
             patch('src.pipeline.setup_logging', return_value=MagicMock()):
            model, metrics = run_pipeline(data_path='dummy.xlsx', model_type='gradient_boosting', compare=False)

        assert model is not None

    @patch('src.pipeline.save_evaluation_report')
    @patch('src.pipeline.generate_model_card', return_value={})
    @patch('src.pipeline.save_model')
    @patch('src.pipeline.save_preprocessor')
    @patch('src.pipeline.load_data')
    def test_pipeline_saves_csv_files(
        self, mock_load, mock_save_prep, mock_save_model,
        mock_model_card, mock_save_report, sample_df, tmp_path
    ):
        mock_load.return_value = sample_df
        with patch('src.pipeline.get_data_path', side_effect=lambda f='', s='': tmp_path / f), \
             patch('src.pipeline.get_model_path', side_effect=lambda f='': tmp_path / f), \
             patch('src.pipeline.MetricsLogger'), \
             patch('src.pipeline.setup_logging', return_value=MagicMock()):
            run_pipeline(data_path='dummy.xlsx', compare=False)

        saved_files = list(tmp_path.iterdir())
        csv_files = [f for f in saved_files if f.suffix == '.csv']
        assert len(csv_files) >= 1
