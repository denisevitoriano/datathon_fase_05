"""
Módulo de treinamento do modelo.
Contém funções para treinar e validar modelos de ML.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from typing import Dict, Any, Tuple, Optional
import joblib
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)


def get_model(model_type: str = 'random_forest', **kwargs) -> Any:
    """
    Retorna instância do modelo especificado.

    Args:
        model_type: Tipo do modelo ('random_forest', 'gradient_boosting')
        **kwargs: Parâmetros adicionais para o modelo

    Returns:
        Instância do modelo
    """
    models = {
        'random_forest': RandomForestClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 10),
            min_samples_split=kwargs.get('min_samples_split', 5),
            min_samples_leaf=kwargs.get('min_samples_leaf', 2),
            random_state=kwargs.get('random_state', 42),
            n_jobs=-1,
            class_weight='balanced'
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 5),
            learning_rate=kwargs.get('learning_rate', 0.1),
            random_state=kwargs.get('random_state', 42)
        ),
    }

    if model_type not in models:
        raise ValueError(f"Modelo '{model_type}' não suportado. Opções: {list(models.keys())}")

    return models[model_type]


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = 'random_forest',
    **kwargs
) -> Any:
    """
    Treina o modelo com os dados fornecidos.

    Args:
        X_train: Features de treino
        y_train: Target de treino
        model_type: Tipo do modelo
        **kwargs: Parâmetros do modelo

    Returns:
        Modelo treinado
    """
    model = get_model(model_type, **kwargs)

    logger.info(f"Treinando modelo {model_type}...")
    model.fit(X_train, y_train)
    logger.info("Modelo treinado com sucesso")

    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Avalia o modelo nos dados de teste.

    Args:
        model: Modelo treinado
        X_test: Features de teste
        y_test: Target de teste

    Returns:
        Dicionário com métricas de avaliação
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }

    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_prob)

    logger.info(f"Métricas de avaliação: {metrics}")
    return metrics


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5
) -> Dict[str, Any]:
    """
    Realiza validação cruzada do modelo.

    Args:
        model: Modelo (não treinado)
        X: Features
        y: Target
        cv: Número de folds

    Returns:
        Dicionário com resultados da validação cruzada
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scoring_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    results = {}

    for metric in scoring_metrics:
        scores = cross_val_score(model, X, y, cv=skf, scoring=metric)
        results[f'{metric}_mean'] = scores.mean()
        results[f'{metric}_std'] = scores.std()

    logger.info(f"Validação cruzada ({cv} folds) concluída")
    return results


def get_feature_importance(
    model: Any,
    feature_names: list
) -> pd.DataFrame:
    """
    Obtém importância das features do modelo.

    Args:
        model: Modelo treinado
        feature_names: Nomes das features

    Returns:
        DataFrame com importância das features ordenado
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    importance_df = pd.DataFrame({
        'feature': feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)

    return importance_df


def train_and_evaluate(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_type: str = 'random_forest',
    **kwargs
) -> Tuple[Any, Dict[str, Any]]:
    """
    Pipeline completo de treino e avaliação com dados já splitados.

    Args:
        X_train: Features de treino (já pré-processadas)
        X_test: Features de teste (já pré-processadas)
        y_train: Target de treino
        y_test: Target de teste
        model_type: Tipo do modelo
        **kwargs: Parâmetros do modelo

    Returns:
        Tupla (modelo treinado, métricas)
    """
    # Treina modelo
    model = train_model(X_train, y_train, model_type, **kwargs)

    # Avalia modelo no conjunto de teste
    metrics = evaluate_model(model, X_test, y_test)

    # Validação cruzada apenas no conjunto de treino
    cv_results = cross_validate_model(get_model(model_type, **kwargs), X_train, y_train)
    metrics['cv_results'] = cv_results

    return model, metrics


def save_model(model: Any, path: str, metadata: Optional[Dict] = None) -> None:
    """
    Salva o modelo e metadados em arquivo.

    Args:
        model: Modelo treinado
        path: Caminho para salvar
        metadata: Metadados opcionais (métricas, data, etc.)
    """
    model_data = {
        'model': model,
        'metadata': metadata or {},
        'saved_at': datetime.now().isoformat()
    }

    joblib.dump(model_data, path)
    logger.info(f"Modelo salvo em {path}")


def load_model(path: str) -> Tuple[Any, Dict]:
    """
    Carrega modelo de arquivo.

    Args:
        path: Caminho do arquivo

    Returns:
        Tupla (modelo, metadados)
    """
    model_data = joblib.load(path)
    logger.info(f"Modelo carregado de {path}")
    return model_data['model'], model_data.get('metadata', {})


def compare_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_types: list = None,
) -> pd.DataFrame:
    """
    Compara performance de diferentes modelos com dados já splitados.

    Args:
        X_train: Features de treino (já pré-processadas)
        X_test: Features de teste (já pré-processadas)
        y_train: Target de treino
        y_test: Target de teste
        model_types: Lista de tipos de modelo a comparar

    Returns:
        DataFrame com comparação de métricas
    """
    if model_types is None:
        model_types = ['random_forest', 'gradient_boosting']

    results = []

    for model_type in model_types:
        logger.info(f"Avaliando {model_type}...")

        model = train_model(X_train, y_train, model_type)
        metrics = evaluate_model(model, X_test, y_test)

        results.append({
            'model': model_type,
            **metrics
        })

    comparison_df = pd.DataFrame(results)
    comparison_df = comparison_df.sort_values('f1', ascending=False)

    return comparison_df
