"""
Testes unitários para o módulo de avaliação.
"""
import pytest
import numpy as np
import pandas as pd

from src.evaluate import (
    generate_classification_report,
    get_confusion_matrix,
    calculate_roc_metrics,
    calculate_precision_recall_metrics,
    find_optimal_threshold,
    evaluate_by_subgroup,
    generate_model_card
)


@pytest.fixture
def sample_predictions():
    """Dados de exemplo para avaliação."""
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0, 1, 0])
    y_prob = np.array([0.2, 0.6, 0.8, 0.9, 0.3, 0.4, 0.75, 0.1, 0.85, 0.25])
    return y_true, y_pred, y_prob


class TestGenerateClassificationReport:
    """Testes para generate_classification_report."""

    def test_returns_string(self, sample_predictions):
        """Verifica se retorna string."""
        y_true, y_pred, _ = sample_predictions
        result = generate_classification_report(y_true, y_pred)
        assert isinstance(result, str)

    def test_contains_metrics(self, sample_predictions):
        """Verifica se contém métricas esperadas."""
        y_true, y_pred, _ = sample_predictions
        result = generate_classification_report(y_true, y_pred)

        assert 'precision' in result
        assert 'recall' in result
        assert 'f1-score' in result

    def test_uses_custom_target_names(self, sample_predictions):
        """Verifica se usa nomes customizados."""
        y_true, y_pred, _ = sample_predictions
        result = generate_classification_report(
            y_true, y_pred,
            target_names=['Negativo', 'Positivo']
        )

        assert 'Negativo' in result
        assert 'Positivo' in result


class TestGetConfusionMatrix:
    """Testes para get_confusion_matrix."""

    def test_returns_dict(self, sample_predictions):
        """Verifica se retorna dicionário."""
        y_true, y_pred, _ = sample_predictions
        result = get_confusion_matrix(y_true, y_pred)

        assert isinstance(result, dict)

    def test_contains_all_values(self, sample_predictions):
        """Verifica se contém todos os valores."""
        y_true, y_pred, _ = sample_predictions
        result = get_confusion_matrix(y_true, y_pred)

        assert 'true_negative' in result
        assert 'false_positive' in result
        assert 'false_negative' in result
        assert 'true_positive' in result

    def test_values_are_integers(self, sample_predictions):
        """Verifica se valores são inteiros."""
        y_true, y_pred, _ = sample_predictions
        result = get_confusion_matrix(y_true, y_pred)

        for value in result.values():
            assert isinstance(value, int)


class TestCalculateRocMetrics:
    """Testes para calculate_roc_metrics."""

    def test_returns_dict(self, sample_predictions):
        """Verifica se retorna dicionário."""
        y_true, _, y_prob = sample_predictions
        result = calculate_roc_metrics(y_true, y_prob)

        assert isinstance(result, dict)

    def test_contains_expected_keys(self, sample_predictions):
        """Verifica se contém chaves esperadas."""
        y_true, _, y_prob = sample_predictions
        result = calculate_roc_metrics(y_true, y_prob)

        assert 'fpr' in result
        assert 'tpr' in result
        assert 'thresholds' in result
        assert 'auc' in result

    def test_auc_in_valid_range(self, sample_predictions):
        """Verifica se AUC está em range válido."""
        y_true, _, y_prob = sample_predictions
        result = calculate_roc_metrics(y_true, y_prob)

        assert 0 <= result['auc'] <= 1


class TestCalculatePrecisionRecallMetrics:
    """Testes para calculate_precision_recall_metrics."""

    def test_returns_dict(self, sample_predictions):
        """Verifica se retorna dicionário."""
        y_true, _, y_prob = sample_predictions
        result = calculate_precision_recall_metrics(y_true, y_prob)

        assert isinstance(result, dict)

    def test_contains_expected_keys(self, sample_predictions):
        """Verifica se contém chaves esperadas."""
        y_true, _, y_prob = sample_predictions
        result = calculate_precision_recall_metrics(y_true, y_prob)

        assert 'precision' in result
        assert 'recall' in result
        assert 'average_precision' in result


class TestFindOptimalThreshold:
    """Testes para find_optimal_threshold."""

    def test_returns_tuple(self, sample_predictions):
        """Verifica se retorna tupla."""
        y_true, _, y_prob = sample_predictions
        result = find_optimal_threshold(y_true, y_prob)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_threshold_in_valid_range(self, sample_predictions):
        """Verifica se threshold está em range válido."""
        y_true, _, y_prob = sample_predictions
        threshold, score = find_optimal_threshold(y_true, y_prob)

        assert 0 < threshold < 1
        assert 0 <= score <= 1

    def test_optimizes_different_metrics(self, sample_predictions):
        """Verifica se otimiza diferentes métricas."""
        y_true, _, y_prob = sample_predictions

        for metric in ['f1', 'precision', 'recall']:
            threshold, _ = find_optimal_threshold(y_true, y_prob, metric=metric)
            assert 0 < threshold < 1


class TestEvaluateBySubgroup:
    """Testes para evaluate_by_subgroup."""

    def test_returns_dataframe(self, sample_predictions):
        """Verifica se retorna DataFrame."""
        y_true, y_pred, _ = sample_predictions
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        })

        result = evaluate_by_subgroup(df, y_true, y_pred, 'group')

        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self, sample_predictions):
        """Verifica se tem colunas esperadas."""
        y_true, y_pred, _ = sample_predictions
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'A', 'B', 'A', 'B', 'A', 'B']
        })

        result = evaluate_by_subgroup(df, y_true, y_pred, 'group')

        assert 'group' in result.columns
        assert 'n_samples' in result.columns
        assert 'accuracy' in result.columns
        assert 'f1' in result.columns


class TestGenerateModelCard:
    """Testes para generate_model_card."""

    def test_returns_dict(self):
        """Verifica se retorna dicionário."""
        result = generate_model_card(
            model_name='test_model',
            metrics={'accuracy': 0.9},
            feature_importance=pd.DataFrame(),
            training_data_info={'n_samples': 100}
        )

        assert isinstance(result, dict)

    def test_contains_expected_fields(self):
        """Verifica se contém campos esperados."""
        result = generate_model_card(
            model_name='test_model',
            metrics={'accuracy': 0.9},
            feature_importance=pd.DataFrame(),
            training_data_info={'n_samples': 100}
        )

        assert 'model_name' in result
        assert 'metrics' in result
        assert 'limitations' in result
        assert 'ethical_considerations' in result
