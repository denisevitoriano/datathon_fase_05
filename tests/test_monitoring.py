"""
Testes unitários para o módulo de monitoramento.
"""
import pytest
import numpy as np
import pandas as pd
import tempfile
import os

from src.monitoring import (
    DriftDetector,
    PredictionLogger,
    PerformanceMonitor
)


@pytest.fixture
def sample_reference_data():
    """Dados de referência para testes."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.normal(5, 1, 100),
        'feature2': np.random.normal(10, 2, 100),
        'feature3': np.random.uniform(0, 1, 100)
    })


@pytest.fixture
def sample_current_data_no_drift(sample_reference_data):
    """Dados atuais sem drift."""
    np.random.seed(43)
    return pd.DataFrame({
        'feature1': np.random.normal(5, 1, 50),
        'feature2': np.random.normal(10, 2, 50),
        'feature3': np.random.uniform(0, 1, 50)
    })


@pytest.fixture
def sample_current_data_with_drift():
    """Dados atuais com drift."""
    np.random.seed(44)
    return pd.DataFrame({
        'feature1': np.random.normal(10, 1, 50),  # Mean shifted from 5 to 10
        'feature2': np.random.normal(20, 2, 50),  # Mean shifted from 10 to 20
        'feature3': np.random.uniform(0.5, 1.5, 50)  # Range shifted
    })


class TestDriftDetector:
    """Testes para DriftDetector."""

    def test_init(self, sample_reference_data):
        """Verifica se inicializa corretamente."""
        detector = DriftDetector(
            sample_reference_data,
            ['feature1', 'feature2']
        )

        assert detector.reference_data is not None
        assert detector.reference_stats is not None

    def test_detect_no_drift(self, sample_reference_data, sample_current_data_no_drift):
        """Verifica se detecta ausência de drift."""
        detector = DriftDetector(
            sample_reference_data,
            ['feature1', 'feature2', 'feature3']
        )

        result = detector.detect_drift(sample_current_data_no_drift)

        assert isinstance(result, dict)
        assert 'drift_detected' in result

    def test_detect_with_drift(self, sample_reference_data, sample_current_data_with_drift):
        """Verifica se detecta presença de drift."""
        detector = DriftDetector(
            sample_reference_data,
            ['feature1', 'feature2']
        )

        result = detector.detect_drift(sample_current_data_with_drift)

        assert result['drift_detected'] is True

    def test_result_contains_features(self, sample_reference_data, sample_current_data_no_drift):
        """Verifica se resultado contém informações por feature."""
        detector = DriftDetector(
            sample_reference_data,
            ['feature1', 'feature2']
        )

        result = detector.detect_drift(sample_current_data_no_drift)

        assert 'features' in result
        assert 'feature1' in result['features']


class TestPredictionLogger:
    """Testes para PredictionLogger."""

    def test_init_creates_log_dir(self):
        """Verifica se cria diretório de logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, 'predictions')
            logger = PredictionLogger(log_dir=log_dir)

            assert os.path.exists(log_dir)

    def test_log_prediction(self):
        """Verifica se registra predição."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(log_dir=tmpdir)
            logger.log_prediction(
                input_data={'feature1': 5},
                prediction=1,
                probability=0.8
            )

            assert len(logger.predictions) == 1

    def test_get_summary(self):
        """Verifica se retorna sumário."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(log_dir=tmpdir)
            logger.log_prediction({'f': 1}, 1, 0.8)
            logger.log_prediction({'f': 2}, 0, 0.3)
            logger.log_prediction({'f': 3}, 1, 0.9)

            summary = logger.get_summary()

            assert summary['total_predictions'] == 3
            assert summary['at_risk_count'] == 2

    def test_save_logs(self):
        """Verifica se salva logs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PredictionLogger(log_dir=tmpdir)
            logger.log_prediction({'f': 1}, 1, 0.8)

            filepath = logger.save_logs()

            assert os.path.exists(filepath)


class TestPerformanceMonitor:
    """Testes para PerformanceMonitor."""

    def test_init(self):
        """Verifica se inicializa corretamente."""
        monitor = PerformanceMonitor()

        assert monitor.predictions == []
        assert monitor.ground_truth == []

    def test_add_prediction(self):
        """Verifica se adiciona predição."""
        monitor = PerformanceMonitor()
        monitor.add_prediction(1, ground_truth=1)

        assert len(monitor.predictions) == 1
        assert len(monitor.ground_truth) == 1

    def test_calculate_metrics(self):
        """Verifica se calcula métricas."""
        monitor = PerformanceMonitor()
        for pred, gt in [(1, 1), (0, 0), (1, 0), (1, 1)]:
            monitor.add_prediction(pred, ground_truth=gt)

        metrics = monitor.calculate_metrics()

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics

    def test_metrics_without_ground_truth(self):
        """Verifica se lida com ausência de ground truth."""
        monitor = PerformanceMonitor()
        monitor.add_prediction(1)
        monitor.add_prediction(0)

        metrics = monitor.calculate_metrics()

        assert metrics == {}

    def test_get_prediction_distribution(self):
        """Verifica se retorna distribuição."""
        monitor = PerformanceMonitor()
        monitor.add_prediction(1)
        monitor.add_prediction(0)
        monitor.add_prediction(1)

        dist = monitor.get_prediction_distribution()

        assert dist['total'] == 3
        assert dist['at_risk'] == 2
        assert dist['not_at_risk'] == 1
