"""
Módulo de monitoramento do modelo.
Contém funções para detectar drift e monitorar performance.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging
from pathlib import Path
from scipy import stats

logger = logging.getLogger(__name__)


class DriftDetector:
    """Classe para detecção de data drift."""

    def __init__(self, reference_data: pd.DataFrame, feature_columns: List[str]):
        """
        Inicializa detector de drift.

        Args:
            reference_data: DataFrame com dados de referência (treino)
            feature_columns: Lista de colunas para monitorar
        """
        self.reference_data = reference_data
        self.feature_columns = feature_columns
        self.reference_stats = self._compute_stats(reference_data)

    def _compute_stats(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calcula estatísticas dos dados.

        Args:
            data: DataFrame

        Returns:
            Dicionário com estatísticas por coluna
        """
        stats_dict = {}

        for col in self.feature_columns:
            if col in data.columns:
                col_data = pd.to_numeric(data[col], errors='coerce').dropna()

                if len(col_data) > 0:
                    stats_dict[col] = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'median': float(col_data.median()),
                        'q1': float(col_data.quantile(0.25)),
                        'q3': float(col_data.quantile(0.75))
                    }

        return stats_dict

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detecta drift entre dados de referência e dados atuais.

        Args:
            current_data: DataFrame com dados atuais
            threshold: Threshold para p-value (default 0.05)

        Returns:
            Dicionário com resultados da detecção
        """
        current_stats = self._compute_stats(current_data)
        drift_results = {
            'timestamp': datetime.now().isoformat(),
            'n_samples_reference': len(self.reference_data),
            'n_samples_current': len(current_data),
            'features': {},
            'drift_detected': False
        }

        drift_count = 0

        for col in self.feature_columns:
            if col in self.reference_data.columns and col in current_data.columns:
                ref_data = pd.to_numeric(self.reference_data[col], errors='coerce').dropna()
                cur_data = pd.to_numeric(current_data[col], errors='coerce').dropna()

                if len(ref_data) > 0 and len(cur_data) > 0:
                    # Kolmogorov-Smirnov test
                    ks_stat, p_value = stats.ks_2samp(ref_data, cur_data)

                    # Population Stability Index (PSI)
                    psi = self._calculate_psi(ref_data.values, cur_data.values)

                    drift_detected = p_value < threshold or psi > 0.2

                    if drift_detected:
                        drift_count += 1

                    drift_results['features'][col] = {
                        'ks_statistic': float(ks_stat),
                        'p_value': float(p_value),
                        'psi': float(psi),
                        'drift_detected': drift_detected,
                        'reference_mean': self.reference_stats.get(col, {}).get('mean'),
                        'current_mean': current_stats.get(col, {}).get('mean')
                    }

        drift_results['drift_detected'] = drift_count > 0
        drift_results['drift_count'] = drift_count
        drift_results['total_features'] = len(self.feature_columns)

        return drift_results

    def _calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calcula Population Stability Index (PSI).

        Args:
            expected: Dados esperados (referência)
            actual: Dados atuais
            n_bins: Número de bins

        Returns:
            Valor do PSI
        """
        # Cria bins baseados nos dados esperados
        breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
        breakpoints = np.unique(breakpoints)

        if len(breakpoints) < 2:
            return 0.0

        # Calcula proporções
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]

        expected_pct = expected_counts / len(expected)
        actual_pct = actual_counts / len(actual)

        # Evita divisão por zero
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)

        # Calcula PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

        return psi


class PredictionLogger:
    """Classe para logging de predições."""

    def __init__(self, log_dir: str = 'logs/predictions'):
        """
        Inicializa logger de predições.

        Args:
            log_dir: Diretório para salvar logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.predictions = []

    def log_prediction(
        self,
        input_data: Dict[str, Any],
        prediction: int,
        probability: float,
        request_id: Optional[str] = None
    ) -> None:
        """
        Registra uma predição.

        Args:
            input_data: Dados de entrada
            prediction: Classe predita
            probability: Probabilidade
            request_id: ID da requisição (opcional)
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'request_id': request_id or str(datetime.now().timestamp()),
            'input': input_data,
            'prediction': prediction,
            'probability': probability
        }

        self.predictions.append(log_entry)

        # Salva periodicamente
        if len(self.predictions) >= 100:
            self.save_logs()

    def save_logs(self) -> str:
        """
        Salva logs em arquivo.

        Returns:
            Caminho do arquivo salvo
        """
        if not self.predictions:
            return ""

        filename = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.log_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.predictions, f, indent=2, ensure_ascii=False)

        self.predictions = []
        logger.info(f"Logs de predições salvos em {filepath}")

        return str(filepath)

    def get_summary(self) -> Dict[str, Any]:
        """
        Retorna sumário das predições.

        Returns:
            Dicionário com estatísticas
        """
        if not self.predictions:
            return {}

        predictions = [p['prediction'] for p in self.predictions]
        probabilities = [p['probability'] for p in self.predictions]

        return {
            'total_predictions': len(predictions),
            'at_risk_count': sum(predictions),
            'at_risk_percentage': sum(predictions) / len(predictions) * 100,
            'avg_probability': np.mean(probabilities),
            'std_probability': np.std(probabilities)
        }


class PerformanceMonitor:
    """Classe para monitoramento de performance do modelo."""

    def __init__(self):
        """Inicializa monitor de performance."""
        self.predictions = []
        self.ground_truth = []
        self.timestamps = []

    def add_prediction(
        self,
        prediction: int,
        ground_truth: Optional[int] = None
    ) -> None:
        """
        Adiciona predição para monitoramento.

        Args:
            prediction: Classe predita
            ground_truth: Valor real (se disponível)
        """
        self.predictions.append(prediction)
        self.ground_truth.append(ground_truth)
        self.timestamps.append(datetime.now())

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas quando ground truth disponível.

        Returns:
            Dicionário com métricas
        """
        # Filtra apenas onde temos ground truth
        valid_indices = [i for i, gt in enumerate(self.ground_truth) if gt is not None]

        if not valid_indices:
            return {}

        y_true = [self.ground_truth[i] for i in valid_indices]
        y_pred = [self.predictions[i] for i in valid_indices]

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'n_samples': len(valid_indices)
        }

    def get_prediction_distribution(self) -> Dict[str, Any]:
        """
        Retorna distribuição das predições.

        Returns:
            Dicionário com distribuição
        """
        if not self.predictions:
            return {}

        total = len(self.predictions)
        at_risk = sum(self.predictions)

        return {
            'total': total,
            'at_risk': at_risk,
            'not_at_risk': total - at_risk,
            'at_risk_rate': at_risk / total if total > 0 else 0
        }


def generate_monitoring_report(
    drift_detector: DriftDetector,
    current_data: pd.DataFrame,
    prediction_logger: PredictionLogger,
    performance_monitor: Optional[PerformanceMonitor] = None
) -> Dict[str, Any]:
    """
    Gera relatório completo de monitoramento.

    Args:
        drift_detector: Detector de drift
        current_data: Dados atuais
        prediction_logger: Logger de predições
        performance_monitor: Monitor de performance (opcional)

    Returns:
        Dicionário com relatório completo
    """
    report = {
        'timestamp': datetime.now().isoformat(),
        'drift_analysis': drift_detector.detect_drift(current_data),
        'prediction_summary': prediction_logger.get_summary()
    }

    if performance_monitor:
        report['performance_metrics'] = performance_monitor.calculate_metrics()
        report['prediction_distribution'] = performance_monitor.get_prediction_distribution()

    return report
