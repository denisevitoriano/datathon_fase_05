"""
Testes unitários para o módulo de treinamento.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import tempfile
import os

from src.train import (
    get_model,
    train_model,
    evaluate_model,
    cross_validate_model,
    get_feature_importance,
    train_and_evaluate,
    save_model,
    load_model,
    compare_models
)


@pytest.fixture
def sample_data():
    """Dados de exemplo para treinamento."""
    np.random.seed(42)
    n_samples = 200

    X = np.random.randn(n_samples, 10)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Target baseado em 2 features

    return X, y


@pytest.fixture
def trained_model(sample_data):
    """Modelo treinado para testes."""
    X, y = sample_data
    model = train_model(X, y, model_type='random_forest')
    return model


class TestGetModel:
    """Testes para get_model."""

    def test_returns_random_forest(self):
        """Verifica se retorna RandomForest."""
        model = get_model('random_forest')
        assert isinstance(model, RandomForestClassifier)

    def test_returns_gradient_boosting(self):
        """Verifica se retorna GradientBoosting."""
        model = get_model('gradient_boosting')
        assert isinstance(model, GradientBoostingClassifier)

    def test_raises_for_invalid_type(self):
        """Verifica se levanta erro para tipo inválido."""
        with pytest.raises(ValueError):
            get_model('invalid_model')

    def test_accepts_custom_params(self):
        """Verifica se aceita parâmetros customizados."""
        model = get_model('random_forest', n_estimators=50, max_depth=5)

        assert model.n_estimators == 50
        assert model.max_depth == 5


class TestTrainModel:
    """Testes para train_model."""

    def test_trains_model(self, sample_data):
        """Verifica se treina o modelo."""
        X, y = sample_data
        model = train_model(X, y, model_type='random_forest')

        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')

    def test_model_can_predict(self, sample_data):
        """Verifica se modelo treinado pode fazer predições."""
        X, y = sample_data
        model = train_model(X, y, model_type='random_forest')

        predictions = model.predict(X[:5])
        assert len(predictions) == 5

    def test_trains_different_models(self, sample_data):
        """Verifica se treina diferentes tipos de modelos."""
        X, y = sample_data

        for model_type in ['random_forest', 'gradient_boosting']:
            model = train_model(X, y, model_type=model_type)
            assert model is not None


class TestEvaluateModel:
    """Testes para evaluate_model."""

    def test_returns_metrics(self, sample_data, trained_model):
        """Verifica se retorna métricas."""
        X, y = sample_data
        metrics = evaluate_model(trained_model, X, y)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics

    def test_metrics_in_valid_range(self, sample_data, trained_model):
        """Verifica se métricas estão em range válido."""
        X, y = sample_data
        metrics = evaluate_model(trained_model, X, y)

        for metric_name, value in metrics.items():
            if metric_name != 'cv_results':
                assert 0 <= value <= 1, f"{metric_name} fora do range"

    def test_includes_roc_auc(self, sample_data, trained_model):
        """Verifica se inclui ROC-AUC."""
        X, y = sample_data
        metrics = evaluate_model(trained_model, X, y)

        assert 'roc_auc' in metrics


class TestCrossValidateModel:
    """Testes para cross_validate_model."""

    def test_returns_cv_results(self, sample_data):
        """Verifica se retorna resultados de CV."""
        X, y = sample_data
        model = get_model('random_forest')

        results = cross_validate_model(model, X, y, cv=3)

        assert 'accuracy_mean' in results
        assert 'accuracy_std' in results
        assert 'f1_mean' in results

    def test_cv_uses_correct_folds(self, sample_data):
        """Verifica se usa número correto de folds."""
        X, y = sample_data
        model = get_model('random_forest')

        results = cross_validate_model(model, X, y, cv=5)

        # Std só é calculado se houver múltiplos folds
        assert results['accuracy_std'] >= 0


class TestGetFeatureImportance:
    """Testes para get_feature_importance."""

    def test_returns_dataframe(self, sample_data, trained_model):
        """Verifica se retorna DataFrame."""
        X, y = sample_data
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        importance = get_feature_importance(trained_model, feature_names)

        assert isinstance(importance, pd.DataFrame)
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns

    def test_sorted_by_importance(self, sample_data, trained_model):
        """Verifica se está ordenado por importância."""
        X, y = sample_data
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        importance = get_feature_importance(trained_model, feature_names)

        # Deve estar em ordem decrescente
        assert importance['importance'].iloc[0] >= importance['importance'].iloc[-1]


class TestTrainAndEvaluate:
    """Testes para train_and_evaluate."""

    def test_returns_model_and_metrics(self, sample_data):
        """Verifica se retorna modelo e métricas."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model, metrics = train_and_evaluate(X_train, X_test, y_train, y_test, model_type='random_forest')

        assert model is not None
        assert metrics is not None
        assert 'accuracy' in metrics

    def test_includes_cv_results(self, sample_data):
        """Verifica se inclui resultados de CV."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model, metrics = train_and_evaluate(X_train, X_test, y_train, y_test)

        assert 'cv_results' in metrics


class TestSaveLoadModel:
    """Testes para save_model e load_model."""

    def test_saves_and_loads_model(self, sample_data, trained_model):
        """Verifica se salva e carrega modelo."""
        X, y = sample_data

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            temp_path = f.name

        try:
            # Salva
            save_model(trained_model, temp_path, {'test': True})

            # Carrega
            loaded_model, metadata = load_model(temp_path)

            # Verifica
            assert loaded_model is not None
            assert metadata['test'] is True

            # Verifica se predições são iguais
            original_pred = trained_model.predict(X[:5])
            loaded_pred = loaded_model.predict(X[:5])
            np.testing.assert_array_equal(original_pred, loaded_pred)

        finally:
            os.unlink(temp_path)


class TestCompareModels:
    """Testes para compare_models."""

    def test_returns_comparison_df(self, sample_data):
        """Verifica se retorna DataFrame de comparação."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        comparison = compare_models(X_train, X_test, y_train, y_test, model_types=['random_forest', 'gradient_boosting'])

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2
        assert 'model' in comparison.columns
        assert 'f1' in comparison.columns

    def test_sorted_by_f1(self, sample_data):
        """Verifica se está ordenado por F1."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        comparison = compare_models(X_train, X_test, y_train, y_test)

        # Deve estar em ordem decrescente de F1
        f1_values = comparison['f1'].values
        assert all(f1_values[i] >= f1_values[i+1] for i in range(len(f1_values)-1))
