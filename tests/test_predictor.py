"""
Testes unitários para o módulo de predição.
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from app.model import predictor
from app.model.predictor import (
    get_model_dir,
    get_risk_level,
    get_confidence,
    is_model_loaded,
    get_model,
    get_preprocessor,
    get_model_metadata,
    get_features_config,
    prepare_input,
)


@pytest.fixture(autouse=True)
def reset_predictor_state():
    """Reseta estado global do predictor antes e depois de cada teste."""
    predictor._model = None
    predictor._preprocessor = None
    predictor._model_metadata = None
    predictor._features_config = None
    yield
    predictor._model = None
    predictor._preprocessor = None
    predictor._model_metadata = None
    predictor._features_config = None


class TestGetModelDir:

    def test_returns_path_object(self):
        result = get_model_dir()
        assert isinstance(result, Path)

    def test_path_ends_with_model(self):
        result = get_model_dir()
        assert result.name == "model"


class TestGetRiskLevel:

    def test_low_risk_below_0_3(self):
        assert get_risk_level(0.0) == "Baixo"
        assert get_risk_level(0.1) == "Baixo"
        assert get_risk_level(0.29) == "Baixo"

    def test_medium_risk_between_0_3_and_0_7(self):
        assert get_risk_level(0.3) == "Médio"
        assert get_risk_level(0.5) == "Médio"
        assert get_risk_level(0.69) == "Médio"

    def test_high_risk_above_0_7(self):
        assert get_risk_level(0.7) == "Alto"
        assert get_risk_level(0.9) == "Alto"
        assert get_risk_level(1.0) == "Alto"


class TestGetConfidence:

    def test_zero_probability_returns_max_confidence(self):
        assert get_confidence(0.0) == pytest.approx(1.0)

    def test_one_probability_returns_max_confidence(self):
        assert get_confidence(1.0) == pytest.approx(1.0)

    def test_midpoint_returns_zero_confidence(self):
        assert get_confidence(0.5) == pytest.approx(0.0)

    def test_0_75_returns_half_confidence(self):
        assert get_confidence(0.75) == pytest.approx(0.5)


class TestIsModelLoaded:

    def test_false_when_both_none(self):
        assert is_model_loaded() is False

    def test_false_when_only_model_set(self):
        predictor._model = MagicMock()
        assert is_model_loaded() is False

    def test_false_when_only_preprocessor_set(self):
        predictor._preprocessor = MagicMock()
        assert is_model_loaded() is False

    def test_true_when_both_set(self):
        predictor._model = MagicMock()
        predictor._preprocessor = MagicMock()
        assert is_model_loaded() is True


class TestGetModelWhenLoaded:

    def test_returns_model_without_reloading(self):
        mock_model = MagicMock()
        predictor._model = mock_model
        result = get_model()
        assert result is mock_model

    def test_returns_preprocessor_without_reloading(self):
        mock_prep = MagicMock()
        predictor._preprocessor = mock_prep
        result = get_preprocessor()
        assert result is mock_prep

    def test_returns_metadata_when_set(self):
        predictor._model_metadata = {"model_type": "random_forest"}
        result = get_model_metadata()
        assert result["model_type"] == "random_forest"

    def test_metadata_returns_empty_dict_when_none_after_load(self):
        def mock_load():
            predictor._model_metadata = None

        with patch.object(predictor, 'load_model_artifacts', side_effect=mock_load):
            predictor._model_metadata = None
            result = get_model_metadata()
        assert result == {}

    def test_returns_features_config_when_set(self):
        predictor._features_config = {"all_features": ["inde", "iaa"]}
        result = get_features_config()
        assert result["all_features"] == ["inde", "iaa"]

    def test_features_config_returns_empty_dict_when_none_after_load(self):
        def mock_load():
            predictor._features_config = None

        with patch.object(predictor, 'load_model_artifacts', side_effect=mock_load):
            predictor._features_config = None
            result = get_features_config()
        assert result == {}


class TestPrepareInput:

    def test_creates_dataframe_with_all_features(self):
        predictor._features_config = {"all_features": ["inde", "iaa", "idade"]}
        df = prepare_input({"inde": 7.5, "iaa": 8.0, "idade": 12})
        assert list(df.columns) == ["inde", "iaa", "idade"]

    def test_fills_missing_features_with_nan(self):
        predictor._features_config = {"all_features": ["inde", "iaa", "idade"]}
        df = prepare_input({"inde": 7.5})
        assert np.isnan(df["iaa"].iloc[0])
        assert np.isnan(df["idade"].iloc[0])

    def test_single_row_dataframe(self):
        predictor._features_config = {"all_features": ["inde", "iaa"]}
        df = prepare_input({"inde": 7.5, "iaa": 8.0})
        assert len(df) == 1

    def test_preserves_provided_values(self):
        predictor._features_config = {"all_features": ["inde", "iaa"]}
        df = prepare_input({"inde": 7.5, "iaa": 8.0})
        assert df["inde"].iloc[0] == pytest.approx(7.5)
        assert df["iaa"].iloc[0] == pytest.approx(8.0)
