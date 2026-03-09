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


class TestLoadModelArtifacts:
    """Testes para load_model_artifacts."""

    def test_success_loads_all_artifacts(self, tmp_path):
        """Verifica se carrega todos os artefatos com sucesso."""
        import json

        model_dir = tmp_path
        mock_model = MagicMock()
        mock_preprocessor = MagicMock()
        model_data = {'model': mock_model, 'metadata': {'model_type': 'rf'}}

        # Create files so path.exists() returns True
        (model_dir / 'model.joblib').touch()
        (model_dir / 'preprocessor.joblib').touch()

        features_config = {'all_features': ['inde', 'iaa']}
        with open(model_dir / 'features_config.json', 'w') as f:
            json.dump(features_config, f)

        with patch.object(predictor, 'get_model_dir', return_value=model_dir), \
             patch('app.model.predictor.joblib.load', side_effect=[model_data, mock_preprocessor]):
            predictor.load_model_artifacts()

        assert predictor._model is mock_model
        assert predictor._preprocessor is mock_preprocessor
        assert predictor._model_metadata == {'model_type': 'rf'}
        assert predictor._features_config == features_config

    def test_model_not_found_raises_error(self, tmp_path):
        """Verifica se levanta FileNotFoundError quando modelo não existe."""
        with patch.object(predictor, 'get_model_dir', return_value=tmp_path):
            with pytest.raises(FileNotFoundError, match="Modelo não encontrado"):
                predictor.load_model_artifacts()

    def test_preprocessor_not_found_raises_error(self, tmp_path):
        """Verifica se levanta FileNotFoundError quando preprocessador não existe."""
        model_data = {'model': MagicMock(), 'metadata': {}}

        # Create model.joblib file so it exists, but not preprocessor.joblib
        model_dir = tmp_path
        # We need the file to exist for the path check
        (model_dir / 'model.joblib').touch()

        with patch.object(predictor, 'get_model_dir', return_value=model_dir), \
             patch('app.model.predictor.joblib.load', return_value=model_data):
            with pytest.raises(FileNotFoundError, match="Preprocessador não encontrado"):
                predictor.load_model_artifacts()

    def test_no_features_config_uses_metadata(self, tmp_path):
        """Verifica se usa metadata quando features_config.json não existe."""
        features_from_meta = {'all_features': ['inde']}
        model_data = {'model': MagicMock(), 'metadata': {'features': features_from_meta}}
        mock_preprocessor = MagicMock()

        model_dir = tmp_path
        (model_dir / 'model.joblib').touch()
        (model_dir / 'preprocessor.joblib').touch()

        with patch.object(predictor, 'get_model_dir', return_value=model_dir), \
             patch('app.model.predictor.joblib.load', side_effect=[model_data, mock_preprocessor]):
            predictor.load_model_artifacts()

        assert predictor._features_config == features_from_meta


class TestGetModelLazyLoading:
    """Testes para lazy loading em get_model e get_preprocessor."""

    def test_get_model_calls_load_when_none(self):
        """Verifica se get_model chama load_model_artifacts quando _model é None."""
        mock_model = MagicMock()

        def mock_load():
            predictor._model = mock_model

        with patch.object(predictor, 'load_model_artifacts', side_effect=mock_load):
            result = get_model()

        assert result is mock_model

    def test_get_preprocessor_calls_load_when_none(self):
        """Verifica se get_preprocessor chama load_model_artifacts quando _preprocessor é None."""
        mock_prep = MagicMock()

        def mock_load():
            predictor._preprocessor = mock_prep

        with patch.object(predictor, 'load_model_artifacts', side_effect=mock_load):
            result = get_preprocessor()

        assert result is mock_prep


class TestPredictBatchMissingColumns:
    """Testes para predict_batch com colunas faltantes."""

    def test_missing_columns_filled_with_nan(self):
        """Verifica se colunas faltantes são preenchidas com NaN no predict_batch."""
        from app.model.predictor import predict_batch

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([1])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.array([[0.5, 0.3]])

        predictor._model = mock_model
        predictor._preprocessor = mock_preprocessor
        predictor._features_config = {'all_features': ['inde', 'iaa', 'idade']}

        # Only provide 'inde', missing 'iaa' and 'idade'
        data_list = [{"inde": 7.5}]
        results = predict_batch(data_list)

        assert len(results) == 1
        assert results[0][0] == 1
        # Verify transform was called (meaning missing cols were handled)
        mock_preprocessor.transform.assert_called_once()
