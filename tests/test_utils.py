"""
Testes unitários para o módulo de utilidades.
"""
import pytest
import logging
import tempfile
import os
from pathlib import Path

from src.utils import (
    setup_logging,
    get_project_root,
    get_data_path,
    get_model_path,
    load_config,
    generate_run_id,
    ensure_dir,
    MetricsLogger
)


class TestSetupLogging:
    """Testes para setup_logging."""

    def test_returns_logger(self):
        """Verifica se retorna logger."""
        result = setup_logging()
        assert isinstance(result, logging.Logger)

    def test_sets_log_level(self):
        """Verifica se define nível de log."""
        logger = setup_logging(log_level=logging.DEBUG)
        assert logger.level == logging.DEBUG


class TestGetProjectRoot:
    """Testes para get_project_root."""

    def test_returns_path(self):
        """Verifica se retorna Path."""
        result = get_project_root()
        assert isinstance(result, Path)

    def test_path_exists(self):
        """Verifica se caminho existe."""
        result = get_project_root()
        assert result.exists()


class TestGetDataPath:
    """Testes para get_data_path."""

    def test_returns_path(self):
        """Verifica se retorna Path."""
        result = get_data_path('test.csv')
        assert isinstance(result, Path)

    def test_includes_subdir(self):
        """Verifica se inclui subdiretório."""
        result = get_data_path('test.csv', subdir='raw')
        assert 'raw' in str(result)

    def test_processed_subdir(self):
        """Verifica se aceita subdir processed."""
        result = get_data_path('test.csv', subdir='processed')
        assert 'processed' in str(result)


class TestGetModelPath:
    """Testes para get_model_path."""

    def test_returns_path(self):
        """Verifica se retorna Path."""
        result = get_model_path('model.joblib')
        assert isinstance(result, Path)

    def test_includes_model_dir(self):
        """Verifica se inclui diretório model."""
        result = get_model_path('test.joblib')
        assert 'model' in str(result)


class TestLoadConfig:
    """Testes para load_config."""

    def test_returns_dict(self):
        """Verifica se retorna dicionário."""
        result = load_config()
        assert isinstance(result, dict)

    def test_has_model_config(self):
        """Verifica se tem configuração de modelo."""
        result = load_config()
        assert 'model' in result

    def test_has_data_config(self):
        """Verifica se tem configuração de dados."""
        result = load_config()
        assert 'data' in result

    def test_has_features_config(self):
        """Verifica se tem configuração de features."""
        result = load_config()
        assert 'features' in result


class TestGenerateRunId:
    """Testes para generate_run_id."""

    def test_returns_string(self):
        """Verifica se retorna string."""
        result = generate_run_id()
        assert isinstance(result, str)

    def test_has_timestamp_format(self):
        """Verifica se tem formato de timestamp."""
        result = generate_run_id()
        assert '_' in result  # YYYYMMDD_HHMMSS


class TestEnsureDir:
    """Testes para ensure_dir."""

    def test_creates_directory(self):
        """Verifica se cria diretório."""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, 'new_subdir')
            result = ensure_dir(new_dir)

            assert result.exists()
            assert result.is_dir()

    def test_returns_path(self):
        """Verifica se retorna Path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = ensure_dir(tmpdir)
            assert isinstance(result, Path)


class TestMetricsLogger:
    """Testes para MetricsLogger."""

    def test_init_creates_log_dir(self):
        """Verifica se cria diretório de logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'test_logs')
            logger = MetricsLogger(log_dir=log_dir)

            assert Path(log_dir).exists()

    def test_log_metrics_stores_entry(self):
        """Verifica se armazena entrada de métricas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            logger.log_metrics({'accuracy': 0.9}, step='training')

            assert len(logger.metrics) == 1

    def test_save_creates_file(self):
        """Verifica se salva arquivo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            logger.log_metrics({'accuracy': 0.9})

            filepath = logger.save()

            assert os.path.exists(filepath)

    def test_save_returns_path(self):
        """Verifica se retorna caminho do arquivo."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(log_dir=tmpdir)
            logger.log_metrics({'f1': 0.85})

            result = logger.save()

            assert isinstance(result, str)
            assert result.endswith('.json')
