"""
Testes unitários para a API.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np

from app.main import app
from app.model import predictor


@pytest.fixture
def client():
    """Cliente de teste para a API."""
    return TestClient(app)


@pytest.fixture
def mock_model():
    """Mock do modelo para testes."""
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.3, 0.7]])
    return model


@pytest.fixture
def mock_preprocessor():
    """Mock do preprocessador para testes."""
    preprocessor = MagicMock()
    preprocessor.transform.return_value = np.array([[0.5, 0.3, 0.2]])
    return preprocessor


@pytest.fixture
def sample_student_input():
    """Dados de exemplo de um estudante."""
    return {
        "iaa": 8.0,
        "ieg": 7.0,
        "ips": 6.5,
        "ipp": 7.5,
        "ida": 6.8,
        "ipv": 7.2,
        "mat": 7.0,
        "por": 6.5,
        "ing": 7.5,
        "idade": 12,
        "ano_ingresso": 2022,
        "genero": "Masculino",
        "instituicao": "Pública"
    }


class TestRootEndpoint:
    """Testes para endpoint raiz."""

    def test_root_returns_200(self, client):
        """Verifica se endpoint raiz retorna 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_api_info(self, client):
        """Verifica se retorna informações da API."""
        response = client.get("/")
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "endpoints" in data


class TestPredictEndpoint:
    """Testes para endpoint /predict."""

    @patch.object(predictor, '_model')
    @patch.object(predictor, '_preprocessor')
    @patch.object(predictor, '_features_config')
    def test_predict_returns_200(self, mock_config, mock_prep, mock_mod,
                                  client, sample_student_input, mock_model, mock_preprocessor):
        """Verifica se endpoint predict retorna 200."""
        # Setup mocks
        predictor._model = mock_model
        predictor._preprocessor = mock_preprocessor
        predictor._features_config = {
            'all_features': ['iaa', 'ieg', 'ips', 'ipp', 'ida', 'ipv',
                            'mat', 'por', 'ing', 'idade', 'ano_ingresso',
                            'genero', 'instituicao']
        }

        response = client.post("/predict", json=sample_student_input)

        # Restaura
        predictor._model = None
        predictor._preprocessor = None
        predictor._features_config = None

        assert response.status_code == 200

    @patch.object(predictor, '_model')
    @patch.object(predictor, '_preprocessor')
    @patch.object(predictor, '_features_config')
    def test_predict_returns_expected_fields(self, mock_config, mock_prep, mock_mod,
                                              client, sample_student_input, mock_model, mock_preprocessor):
        """Verifica se retorna campos esperados."""
        predictor._model = mock_model
        predictor._preprocessor = mock_preprocessor
        predictor._features_config = {
            'all_features': ['iaa', 'ieg', 'ips', 'ipp', 'ida', 'ipv',
                            'mat', 'por', 'ing', 'idade', 'ano_ingresso',
                            'genero', 'instituicao']
        }

        response = client.post("/predict", json=sample_student_input)
        data = response.json()

        predictor._model = None
        predictor._preprocessor = None
        predictor._features_config = None

        assert "at_risk" in data
        assert "risk_probability" in data
        assert "risk_level" in data
        assert "confidence" in data

    def test_predict_validates_input(self, client):
        """Verifica se valida entrada inválida."""
        invalid_input = {
            "iaa": 15.0,  # Valor fora do range (0-10)
        }

        response = client.post("/predict", json=invalid_input)

        # Deve retornar erro de validação
        assert response.status_code == 422


class TestPredictBatchEndpoint:
    """Testes para endpoint /predict/batch."""

    @patch.object(predictor, '_model')
    @patch.object(predictor, '_preprocessor')
    @patch.object(predictor, '_features_config')
    def test_batch_predict_returns_200(self, mock_config, mock_prep, mock_mod,
                                        client, sample_student_input, mock_model, mock_preprocessor):
        """Verifica se endpoint batch retorna 200."""
        mock_model.predict.return_value = np.array([1, 0])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        mock_preprocessor.transform.return_value = np.array([[0.5, 0.3], [0.4, 0.6]])

        predictor._model = mock_model
        predictor._preprocessor = mock_preprocessor
        predictor._features_config = {
            'all_features': ['iaa', 'ieg', 'ips', 'ipp', 'ida', 'ipv',
                            'mat', 'por', 'ing', 'idade', 'ano_ingresso',
                            'genero', 'instituicao']
        }

        batch_input = {"students": [sample_student_input, sample_student_input]}
        response = client.post("/predict/batch", json=batch_input)

        predictor._model = None
        predictor._preprocessor = None
        predictor._features_config = None

        assert response.status_code == 200

    @patch.object(predictor, '_model')
    @patch.object(predictor, '_preprocessor')
    @patch.object(predictor, '_features_config')
    def test_batch_returns_summary(self, mock_config, mock_prep, mock_mod,
                                    client, sample_student_input, mock_model, mock_preprocessor):
        """Verifica se retorna sumário do batch."""
        mock_model.predict.return_value = np.array([1, 0])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
        mock_preprocessor.transform.return_value = np.array([[0.5, 0.3], [0.4, 0.6]])

        predictor._model = mock_model
        predictor._preprocessor = mock_preprocessor
        predictor._features_config = {
            'all_features': ['iaa', 'ieg', 'ips', 'ipp', 'ida', 'ipv',
                            'mat', 'por', 'ing', 'idade', 'ano_ingresso',
                            'genero', 'instituicao']
        }

        batch_input = {"students": [sample_student_input, sample_student_input]}
        response = client.post("/predict/batch", json=batch_input)
        data = response.json()

        predictor._model = None
        predictor._preprocessor = None
        predictor._features_config = None

        assert "predictions" in data
        assert "total_processed" in data
        assert "at_risk_count" in data
        assert "at_risk_percentage" in data


class TestHealthEndpoint:
    """Testes para endpoint /health."""

    def test_health_returns_200(self, client):
        """Verifica se endpoint health retorna 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        """Verifica se retorna status."""
        response = client.get("/health")
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data


class TestModelInfoEndpoint:
    """Testes para endpoint /model/info."""

    @patch.object(predictor, '_model')
    @patch.object(predictor, '_preprocessor')
    @patch.object(predictor, '_model_metadata')
    @patch.object(predictor, '_features_config')
    def test_model_info_returns_200_when_loaded(self, mock_config, mock_metadata, mock_prep, mock_mod, client):
        """Verifica se retorna 200 quando modelo carregado."""
        predictor._model = MagicMock()
        predictor._preprocessor = MagicMock()
        predictor._model_metadata = {'model_type': 'random_forest', 'metrics': {}}
        predictor._features_config = {'all_features': [], 'numeric_features': [], 'categorical_features': []}

        response = client.get("/model/info")

        predictor._model = None
        predictor._preprocessor = None
        predictor._model_metadata = None
        predictor._features_config = None

        assert response.status_code == 200


class TestFeaturesEndpoint:
    """Testes para endpoint /features."""

    @patch.object(predictor, '_features_config')
    def test_features_returns_200(self, mock_config, client):
        """Verifica se endpoint features retorna 200."""
        predictor._features_config = {
            'numeric_features': ['iaa', 'idade'],
            'categorical_features': ['genero'],
            'all_features': ['iaa', 'idade', 'genero']
        }

        response = client.get("/features")

        predictor._features_config = None

        assert response.status_code == 200

    @patch.object(predictor, '_features_config')
    def test_features_returns_feature_lists(self, mock_config, client):
        """Verifica se retorna listas de features."""
        predictor._features_config = {
            'numeric_features': ['iaa', 'idade'],
            'categorical_features': ['genero'],
            'all_features': ['iaa', 'idade', 'genero']
        }

        response = client.get("/features")
        data = response.json()

        predictor._features_config = None

        assert "numeric_features" in data
        assert "categorical_features" in data
        assert "total" in data
