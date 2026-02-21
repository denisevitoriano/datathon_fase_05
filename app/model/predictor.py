"""
Módulo de predição.
Carrega modelo e faz predições.
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Variáveis globais para armazenar modelo e preprocessador
_model = None
_preprocessor = None
_model_metadata = None
_features_config = None


def get_model_dir() -> Path:
    """Retorna diretório do modelo."""
    return Path(__file__).parent


def load_model_artifacts() -> None:
    """Carrega modelo, preprocessador e configurações."""
    global _model, _preprocessor, _model_metadata, _features_config

    model_dir = get_model_dir()

    # Carrega modelo
    model_path = model_dir / 'model.joblib'
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}")

    model_data = joblib.load(model_path)
    _model = model_data['model']
    _model_metadata = model_data.get('metadata', {})

    # Carrega preprocessador
    preprocessor_path = model_dir / 'preprocessor.joblib'
    if not preprocessor_path.exists():
        raise FileNotFoundError(f"Preprocessador não encontrado em {preprocessor_path}")

    _preprocessor = joblib.load(preprocessor_path)

    # Carrega configuração de features
    features_path = model_dir / 'features_config.json'
    if features_path.exists():
        with open(features_path, 'r') as f:
            _features_config = json.load(f)
    else:
        _features_config = _model_metadata.get('features', {})

    logger.info("Artefatos do modelo carregados com sucesso")


def get_model():
    """Retorna modelo carregado."""
    if _model is None:
        load_model_artifacts()
    return _model


def get_preprocessor():
    """Retorna preprocessador carregado."""
    if _preprocessor is None:
        load_model_artifacts()
    return _preprocessor


def get_model_metadata() -> Dict[str, Any]:
    """Retorna metadados do modelo."""
    if _model_metadata is None:
        load_model_artifacts()
    return _model_metadata or {}


def get_features_config() -> Dict[str, Any]:
    """Retorna configuração de features."""
    if _features_config is None:
        load_model_artifacts()
    return _features_config or {}


def prepare_input(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Prepara dados de entrada para predição.

    Args:
        data: Dicionário com dados do estudante

    Returns:
        DataFrame preparado
    """
    features_config = get_features_config()
    all_features = features_config.get('all_features', [])

    # Cria DataFrame com todas as features esperadas
    df = pd.DataFrame([data])

    # Garante que todas as colunas existem
    for col in all_features:
        if col not in df.columns:
            df[col] = np.nan

    # Seleciona apenas as features esperadas na ordem correta
    df = df[all_features]

    return df


def predict(data: Dict[str, Any]) -> Tuple[int, float]:
    """
    Faz predição para um estudante.

    Args:
        data: Dicionário com dados do estudante

    Returns:
        Tupla (classe_predita, probabilidade)
    """
    model = get_model()
    preprocessor = get_preprocessor()

    # Prepara entrada
    df = prepare_input(data)

    # Aplica preprocessamento
    X = preprocessor.transform(df)

    # Faz predição
    prediction = model.predict(X)[0]
    probability = model.predict_proba(X)[0]

    # Probabilidade da classe positiva (em risco)
    risk_probability = probability[1]

    return int(prediction), float(risk_probability)


def predict_batch(data_list: list) -> list:
    """
    Faz predição para múltiplos estudantes.

    Args:
        data_list: Lista de dicionários com dados dos estudantes

    Returns:
        Lista de tuplas (classe_predita, probabilidade)
    """
    model = get_model()
    preprocessor = get_preprocessor()
    features_config = get_features_config()
    all_features = features_config.get('all_features', [])

    # Prepara todos os dados
    dfs = []
    for data in data_list:
        df = pd.DataFrame([data])
        for col in all_features:
            if col not in df.columns:
                df[col] = np.nan
        df = df[all_features]
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # Aplica preprocessamento
    X = preprocessor.transform(combined_df)

    # Faz predições
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return [(int(p), float(prob)) for p, prob in zip(predictions, probabilities)]


def get_risk_level(probability: float) -> str:
    """
    Converte probabilidade em nível de risco.

    Args:
        probability: Probabilidade de risco (0-1)

    Returns:
        Nível de risco (Baixo/Médio/Alto)
    """
    if probability < 0.3:
        return "Baixo"
    elif probability < 0.7:
        return "Médio"
    else:
        return "Alto"


def get_confidence(probability: float) -> float:
    """
    Calcula confiança da predição.

    Args:
        probability: Probabilidade de risco

    Returns:
        Confiança (quão longe de 0.5)
    """
    return abs(probability - 0.5) * 2


def is_model_loaded() -> bool:
    """Verifica se modelo está carregado."""
    return _model is not None and _preprocessor is not None
