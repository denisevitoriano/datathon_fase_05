"""
Testes unitários para as rotas de monitoramento.
"""
import pytest
import json
from unittest.mock import patch, mock_open
from fastapi.testclient import TestClient

from app.main import app
import app.monitoring_routes as monitoring_routes


@pytest.fixture(autouse=True)
def reset_state():
    """Reseta estado global antes e depois de cada teste."""
    monitoring_routes._prediction_history.clear()
    monitoring_routes._drift_reports.clear()
    yield
    monitoring_routes._prediction_history.clear()
    monitoring_routes._drift_reports.clear()


@pytest.fixture
def client():
    return TestClient(app)


class TestDashboardEndpoint:

    def test_dashboard_returns_200(self, client):
        response = client.get("/monitoring/dashboard")
        assert response.status_code == 200

    def test_dashboard_empty_returns_zero_totals(self, client):
        response = client.get("/monitoring/dashboard")
        data = response.json()
        assert data["prediction_summary"]["total"] == 0
        assert data["prediction_summary"]["at_risk_rate"] == 0

    def test_dashboard_with_predictions(self, client):
        monitoring_routes._prediction_history.extend([
            {"prediction": 1},
            {"prediction": 0},
        ])
        response = client.get("/monitoring/dashboard")
        data = response.json()
        assert data["prediction_summary"]["total"] == 2
        assert data["prediction_summary"]["at_risk"] == 1
        assert data["prediction_summary"]["at_risk_rate"] == 0.5

    def test_dashboard_returns_expected_keys(self, client):
        response = client.get("/monitoring/dashboard")
        data = response.json()
        assert "timestamp" in data
        assert "prediction_summary" in data
        assert "drift_reports_count" in data


class TestLogPredictionEndpoint:

    def test_log_prediction_returns_200(self, client):
        response = client.post("/monitoring/log-prediction", json={"prediction": 1})
        assert response.status_code == 200

    def test_log_prediction_stores_entry(self, client):
        client.post("/monitoring/log-prediction", json={"prediction": 1, "probability": 0.8})
        assert len(monitoring_routes._prediction_history) == 1

    def test_log_prediction_returns_count(self, client):
        response = client.post("/monitoring/log-prediction", json={"prediction": 0})
        data = response.json()
        assert data["status"] == "logged"
        assert data["total_predictions"] == 1

    def test_log_multiple_predictions(self, client):
        client.post("/monitoring/log-prediction", json={"prediction": 0})
        client.post("/monitoring/log-prediction", json={"prediction": 1})
        assert len(monitoring_routes._prediction_history) == 2

    def test_log_prediction_limits_to_10000(self, client):
        monitoring_routes._prediction_history.extend([{"prediction": 0}] * 10000)
        client.post("/monitoring/log-prediction", json={"prediction": 1})
        assert len(monitoring_routes._prediction_history) == 10000


class TestPredictionHistoryEndpoint:

    def test_returns_200(self, client):
        response = client.get("/monitoring/prediction-history")
        assert response.status_code == 200

    def test_returns_expected_keys(self, client):
        response = client.get("/monitoring/prediction-history")
        data = response.json()
        assert "total" in data
        assert "limit" in data
        assert "predictions" in data

    def test_respects_limit_param(self, client):
        monitoring_routes._prediction_history.extend([{"prediction": 0}] * 50)
        response = client.get("/monitoring/prediction-history?limit=10")
        data = response.json()
        assert len(data["predictions"]) == 10

    def test_returns_empty_list_when_no_history(self, client):
        response = client.get("/monitoring/prediction-history")
        data = response.json()
        assert data["predictions"] == []
        assert data["total"] == 0


class TestDriftStatusEndpoint:

    def test_no_reports_returns_no_reports_status(self, client):
        response = client.get("/monitoring/drift-status")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_reports"

    def test_drift_detected_status(self, client):
        monitoring_routes._drift_reports.append({
            "drift_detected": True,
            "timestamp": "2026-01-01",
            "drift_count": 3,
            "total_features": 10
        })
        response = client.get("/monitoring/drift-status")
        data = response.json()
        assert data["status"] == "drift_detected"
        assert data["features_with_drift"] == 3

    def test_no_drift_status(self, client):
        monitoring_routes._drift_reports.append({
            "drift_detected": False,
            "timestamp": "2026-01-01",
            "drift_count": 0,
            "total_features": 10
        })
        response = client.get("/monitoring/drift-status")
        data = response.json()
        assert data["status"] == "no_drift"

    def test_returns_last_report_info(self, client):
        monitoring_routes._drift_reports.append({
            "drift_detected": True,
            "timestamp": "2026-01-01",
            "drift_count": 2,
            "total_features": 5
        })
        response = client.get("/monitoring/drift-status")
        data = response.json()
        assert data["total_features"] == 5


class TestMetricsSummaryEndpoint:

    def test_no_metrics_file_returns_no_metrics(self, client):
        with patch("pathlib.Path.exists", return_value=False):
            response = client.get("/monitoring/metrics-summary")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "no_metrics"

    def test_with_metrics_file_returns_model_info(self, client):
        model_card = {
            "model_name": "random_forest",
            "metrics": {"accuracy": 0.85, "f1": 0.80},
            "training_data": {"n_samples": 1000}
        }
        with patch("pathlib.Path.exists", return_value=True), \
             patch("builtins.open", mock_open(read_data=json.dumps(model_card))), \
             patch("json.load", return_value=model_card):
            response = client.get("/monitoring/metrics-summary")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
