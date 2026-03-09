"""
Métricas Prometheus para monitoramento do modelo.
"""
from prometheus_client import Counter, Gauge, Histogram

# --- Predições ---
PREDICTIONS_TOTAL = Counter(
    'model_predictions_total',
    'Total de predições realizadas',
    ['risk_level']
)

PREDICTION_PROBABILITY = Histogram(
    'model_prediction_probability',
    'Distribuição das probabilidades de risco',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

PREDICTION_LATENCY = Histogram(
    'model_prediction_latency_seconds',
    'Tempo de inferência do modelo',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

BATCH_SIZE = Histogram(
    'model_batch_size',
    'Tamanho dos lotes de predição',
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

AT_RISK_RATE = Gauge(
    'model_at_risk_rate',
    'Taxa atual de estudantes em risco (últimas predições)'
)

# --- Drift ---
DRIFT_DETECTED = Gauge(
    'model_drift_detected',
    'Indica se drift foi detectado (1=sim, 0=não)'
)

DRIFT_FEATURES_COUNT = Gauge(
    'model_drift_features_count',
    'Número de features com drift detectado'
)

DRIFT_TOTAL_FEATURES = Gauge(
    'model_drift_total_features',
    'Total de features monitoradas'
)
