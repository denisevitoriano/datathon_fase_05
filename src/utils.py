"""
Módulo de utilidades.
Contém funções auxiliares para logging, configurações e helpers.
"""
import logging
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import os


def setup_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    """
    Configura logging para a aplicação.

    Args:
        log_level: Nível de log (default: INFO)
        log_file: Arquivo para salvar logs (opcional)
        log_format: Formato do log (opcional)

    Returns:
        Logger configurado
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    # Configura handler para console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))

    # Configura logger raiz
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Adiciona handler para arquivo se especificado
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)

    return root_logger


def get_project_root() -> Path:
    """
    Retorna o diretório raiz do projeto.

    Returns:
        Path do diretório raiz
    """
    return Path(__file__).parent.parent


def get_data_path(filename: str = '', subdir: str = 'raw') -> Path:
    """
    Retorna caminho para arquivo de dados.

    Args:
        filename: Nome do arquivo
        subdir: Subdiretório ('raw' ou 'processed')

    Returns:
        Path completo
    """
    return get_project_root() / 'data' / subdir / filename


def get_model_path(filename: str = '') -> Path:
    """
    Retorna caminho para arquivo de modelo.

    Args:
        filename: Nome do arquivo

    Returns:
        Path completo
    """
    path = get_project_root() / 'app' / 'model'
    path.mkdir(parents=True, exist_ok=True)
    return path / filename


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Carrega configurações do projeto.

    Args:
        config_path: Caminho para arquivo de configuração (opcional)

    Returns:
        Dicionário com configurações
    """
    default_config = {
        'model': {
            'type': 'random_forest',
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }
        },
        'data': {
            'test_size': 0.2,
            'random_state': 42
        },
        'features': {
            'numeric': [
                'iaa', 'ieg', 'ips', 'ipp', 'ida', 'ipv',
                'mat', 'por', 'ing', 'idade'
            ],
            'categorical': ['genero', 'instituicao']
        }
    }

    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
            # Merge configs
            default_config.update(user_config)

    return default_config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Salva configurações em arquivo.

    Args:
        config: Dicionário com configurações
        config_path: Caminho para salvar
    """
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def generate_run_id() -> str:
    """
    Gera ID único para execução.

    Returns:
        String com ID único
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def ensure_dir(path: str) -> Path:
    """
    Garante que diretório existe.

    Args:
        path: Caminho do diretório

    Returns:
        Path do diretório
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class MetricsLogger:
    """Classe para logging estruturado de métricas."""

    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = ensure_dir(log_dir)
        self.run_id = generate_run_id()
        self.metrics = []

    def log_metrics(self, metrics: Dict[str, Any], step: str = 'training') -> None:
        """
        Registra métricas.

        Args:
            metrics: Dicionário com métricas
            step: Etapa do processo
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'run_id': self.run_id,
            'step': step,
            'metrics': metrics
        }
        self.metrics.append(entry)

    def save(self) -> str:
        """
        Salva métricas em arquivo.

        Returns:
            Caminho do arquivo salvo
        """
        filepath = self.log_dir / f'metrics_{self.run_id}.json'
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False, default=str)
        return str(filepath)
