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


class TestSetupLoggingWithFile:
    """Testes para setup_logging com arquivo de log."""

    def test_creates_log_file(self):
        """Verifica se cria arquivo de log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, 'subdir', 'test.log')
            logger = setup_logging(log_file=log_file)

            logger.info("Mensagem de teste")

            assert os.path.exists(log_file)

    def test_creates_parent_directory(self):
        """Verifica se cria diretório pai do arquivo de log."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, 'nested', 'dir', 'test.log')
            setup_logging(log_file=log_file)

            assert os.path.exists(os.path.join(tmpdir, 'nested', 'dir'))

    def test_file_handler_added(self):
        """Verifica se file handler é adicionado ao logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = os.path.join(tmpdir, 'test.log')
            logger = setup_logging(log_file=log_file)

            file_handlers = [
                h for h in logger.handlers
                if isinstance(h, logging.FileHandler)
            ]
            assert len(file_handlers) >= 1


class TestLoadConfigWithFile:
    """Testes para load_config com arquivo de configuração."""

    def test_loads_config_from_file(self):
        """Verifica se carrega configuração de arquivo."""
        import json as json_mod
        custom_config = {"model": {"type": "xgboost"}}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_mod.dump(custom_config, f)
            config_path = f.name

        try:
            result = load_config(config_path=config_path)
            assert result['model']['type'] == 'xgboost'
        finally:
            os.unlink(config_path)

    def test_merges_with_default_config(self):
        """Verifica se faz merge com configuração padrão."""
        import json as json_mod
        custom_config = {"custom_key": "custom_value"}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_mod.dump(custom_config, f)
            config_path = f.name

        try:
            result = load_config(config_path=config_path)
            assert 'custom_key' in result
            assert 'data' in result  # Default key still present
        finally:
            os.unlink(config_path)

    def test_nonexistent_path_returns_default(self):
        """Verifica se retorna padrão quando caminho não existe."""
        result = load_config(config_path='/tmp/nonexistent_config_12345.json')
        assert 'model' in result
        assert result['model']['type'] == 'random_forest'


class TestSaveConfig:
    """Testes para save_config."""

    def test_saves_config_to_file(self):
        """Verifica se salva configuração em arquivo."""
        from src.utils import save_config
        import json as json_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.json')
            config = {"model": {"type": "xgboost"}}

            save_config(config, config_path)

            assert os.path.exists(config_path)
            with open(config_path, 'r') as f:
                saved = json_mod.load(f)
            assert saved['model']['type'] == 'xgboost'

    def test_creates_parent_directories(self):
        """Verifica se cria diretórios pai."""
        from src.utils import save_config

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'nested', 'dir', 'config.json')
            save_config({"key": "value"}, config_path)

            assert os.path.exists(config_path)

    def test_saves_with_utf8_encoding(self):
        """Verifica se salva com encoding UTF-8."""
        from src.utils import save_config
        import json as json_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, 'config.json')
            config = {"descrição": "configuração com acentos"}

            save_config(config, config_path)

            with open(config_path, 'r', encoding='utf-8') as f:
                saved = json_mod.load(f)
            assert saved['descrição'] == 'configuração com acentos'
