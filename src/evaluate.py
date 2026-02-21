"""
Módulo de avaliação do modelo.
Contém funções para avaliação detalhada e geração de relatórios.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from typing import Dict, Any, Tuple
import logging
import json

logger = logging.getLogger(__name__)


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list = None
) -> str:
    """
    Gera relatório de classificação detalhado.

    Args:
        y_true: Valores verdadeiros
        y_pred: Valores preditos
        target_names: Nomes das classes

    Returns:
        Relatório em formato string
    """
    if target_names is None:
        target_names = ['Sem Risco', 'Em Risco']

    report = classification_report(y_true, y_pred, target_names=target_names)
    return report


def get_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, int]:
    """
    Calcula matriz de confusão.

    Args:
        y_true: Valores verdadeiros
        y_pred: Valores preditos

    Returns:
        Dicionário com valores da matriz
    """
    cm = confusion_matrix(y_true, y_pred)

    return {
        'true_negative': int(cm[0, 0]),
        'false_positive': int(cm[0, 1]),
        'false_negative': int(cm[1, 0]),
        'true_positive': int(cm[1, 1])
    }


def calculate_roc_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Calcula métricas da curva ROC.

    Args:
        y_true: Valores verdadeiros
        y_prob: Probabilidades preditas

    Returns:
        Dicionário com FPR, TPR e AUC
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'thresholds': thresholds.tolist(),
        'auc': roc_auc
    }


def calculate_precision_recall_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, Any]:
    """
    Calcula métricas de precisão-recall.

    Args:
        y_true: Valores verdadeiros
        y_prob: Probabilidades preditas

    Returns:
        Dicionário com precision, recall e average precision
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)

    return {
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'thresholds': thresholds.tolist(),
        'average_precision': avg_precision
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Encontra threshold ótimo para maximizar métrica.

    Args:
        y_true: Valores verdadeiros
        y_prob: Probabilidades preditas
        metric: Métrica a otimizar ('f1', 'recall', 'precision')

    Returns:
        Tupla (threshold_ótimo, valor_da_métrica)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0

    metric_funcs = {
        'f1': f1_score,
        'precision': precision_score,
        'recall': recall_score
    }

    metric_func = metric_funcs.get(metric, f1_score)

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = metric_func(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    logger.info(f"Threshold ótimo para {metric}: {best_threshold:.2f} (score: {best_score:.4f})")
    return best_threshold, best_score


def evaluate_by_subgroup(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_col: str
) -> pd.DataFrame:
    """
    Avalia performance por subgrupo.

    Args:
        df: DataFrame original com coluna de grupo
        y_true: Valores verdadeiros
        y_pred: Valores preditos
        group_col: Nome da coluna de agrupamento

    Returns:
        DataFrame com métricas por grupo
    """
    from sklearn.metrics import accuracy_score, f1_score

    results = []

    for group in df[group_col].unique():
        mask = df[group_col] == group
        if mask.sum() > 0:
            y_true_group = y_true[mask]
            y_pred_group = y_pred[mask]

            results.append({
                'group': group,
                'n_samples': int(mask.sum()),
                'accuracy': accuracy_score(y_true_group, y_pred_group),
                'f1': f1_score(y_true_group, y_pred_group, zero_division=0)
            })

    return pd.DataFrame(results)


def generate_model_card(
    model_name: str,
    metrics: Dict[str, float],
    feature_importance: pd.DataFrame,
    training_data_info: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Gera um Model Card com informações do modelo.

    Args:
        model_name: Nome do modelo
        metrics: Métricas de performance
        feature_importance: DataFrame com importância das features
        training_data_info: Informações sobre dados de treino

    Returns:
        Dicionário com Model Card
    """
    model_card = {
        'model_name': model_name,
        'model_description': 'Modelo para prever risco de defasagem escolar de estudantes',
        'intended_use': 'Identificar estudantes em risco para intervenção precoce',
        'metrics': metrics,
        'top_features': feature_importance.head(10).to_dict('records') if not feature_importance.empty else [],
        'training_data': training_data_info,
        'limitations': [
            'Modelo treinado com dados de uma única organização (Passos Mágicos)',
            'Performance pode variar para estudantes com características diferentes',
            'Requer atualização periódica para manter relevância'
        ],
        'ethical_considerations': [
            'Previsões não devem ser usadas como única base para decisões',
            'Recomenda-se análise complementar por profissionais',
            'Modelo visa auxiliar, não substituir, avaliação humana'
        ]
    }

    return model_card


def save_evaluation_report(
    metrics: Dict[str, Any],
    output_path: str
) -> None:
    """
    Salva relatório de avaliação em JSON.

    Args:
        metrics: Dicionário com métricas
        output_path: Caminho para salvar
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"Relatório de avaliação salvo em {output_path}")
