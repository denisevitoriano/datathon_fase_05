"""
Pipeline principal de treinamento.
Script para executar todo o fluxo de ML.
"""
import sys
from pathlib import Path

# Adiciona diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import logging
import json

from src.preprocessing import (
    load_data, standardize_column_names, clean_data,
    create_target, remove_leaky_features, split_features_target
)
from src.feature_engineering import (
    prepare_features_for_training, fit_preprocessor,
    save_preprocessor, get_feature_names
)
from src.train import (
    train_and_evaluate, save_model, compare_models, get_feature_importance
)
from src.evaluate import (
    generate_classification_report, get_confusion_matrix,
    generate_model_card, save_evaluation_report
)
from src.utils import (
    setup_logging, get_data_path, get_model_path,
    MetricsLogger, load_config
)


def run_pipeline(
    data_path: str = None,
    model_type: str = 'random_forest',
    compare: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Executa pipeline completa de treinamento.

    Args:
        data_path: Caminho para arquivo de dados
        model_type: Tipo de modelo a treinar
        compare: Se True, compara diferentes modelos

    Returns:
        Tupla (modelo_treinado, métricas)
    """
    # Setup
    logger = setup_logging(log_file='logs/training.log')
    metrics_logger = MetricsLogger()
    config = load_config()

    logger.info("="*60)
    logger.info("Iniciando pipeline de treinamento")
    logger.info("="*60)

    # 1. Carrega dados
    if data_path is None:
        data_path = get_data_path('BASE DE DADOS PEDE 2024 - DATATHON.xlsx', 'raw')

    logger.info(f"Carregando dados de {data_path}")
    df = load_data(str(data_path), sheet_name='PEDE2024')

    # 2. Pré-processamento
    logger.info("Executando pré-processamento...")
    df = standardize_column_names(df)
    df = clean_data(df)
    df = create_target(df, binary=True)
    df = remove_leaky_features(df)

    # 3. Feature Engineering
    logger.info("Executando feature engineering...")
    df, numeric_features, categorical_features = prepare_features_for_training(df)

    # Filtra apenas features que existem no DataFrame
    numeric_features = [f for f in numeric_features if f in df.columns]
    categorical_features = [f for f in categorical_features if f in df.columns]

    logger.info(f"Features numéricas: {numeric_features}")
    logger.info(f"Features categóricas: {categorical_features}")

    # 4. Prepara dados para treinamento
    all_features = numeric_features + categorical_features
    X = df[all_features].copy()
    y = df['target'].copy()

    logger.info(f"Shape dos dados: X={X.shape}, y={y.shape}")
    logger.info(f"Distribuição do target: {y.value_counts().to_dict()}")

    # 5. Fit preprocessor e transforma dados
    preprocessor, X_transformed = fit_preprocessor(X, numeric_features, categorical_features)

    # 6. Compara modelos (opcional)
    if compare:
        logger.info("Comparando modelos...")
        comparison = compare_models(X_transformed, y.values)
        logger.info(f"\nComparação de modelos:\n{comparison.to_string()}")
        metrics_logger.log_metrics(comparison.to_dict('records'), 'model_comparison')

        # Escolhe melhor modelo baseado em F1
        best_model_type = comparison.iloc[0]['model']
        logger.info(f"Melhor modelo: {best_model_type}")
        model_type = best_model_type

    # 7. Treina modelo final
    logger.info(f"Treinando modelo final: {model_type}")
    model, metrics = train_and_evaluate(
        X_transformed, y.values,
        model_type=model_type,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )

    # 8. Feature importance
    feature_names = get_feature_names(preprocessor)
    importance_df = get_feature_importance(model, feature_names)
    if not importance_df.empty:
        logger.info(f"\nTop 10 features mais importantes:\n{importance_df.head(10).to_string()}")

    # 9. Salva artefatos
    logger.info("Salvando artefatos...")

    # Salva modelo
    model_metadata = {
        'model_type': model_type,
        'metrics': metrics,
        'features': {
            'numeric': numeric_features,
            'categorical': categorical_features
        },
        'feature_names': feature_names
    }
    save_model(model, str(get_model_path('model.joblib')), model_metadata)

    # Salva preprocessor
    save_preprocessor(preprocessor, str(get_model_path('preprocessor.joblib')))

    # Salva configuração de features
    features_config = {
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'all_features': all_features
    }
    with open(get_model_path('features_config.json'), 'w') as f:
        json.dump(features_config, f, indent=2)

    # 10. Gera relatórios
    logger.info("Gerando relatórios...")

    # Model card
    training_data_info = {
        'source': 'Passos Mágicos - PEDE 2024',
        'n_samples': len(df),
        'n_features': len(all_features),
        'target_distribution': y.value_counts().to_dict()
    }
    model_card = generate_model_card(model_type, metrics, importance_df, training_data_info)
    save_evaluation_report(model_card, 'logs/model_card.json')

    # Métricas
    metrics_logger.log_metrics(metrics, 'final_evaluation')
    metrics_logger.save()

    logger.info("="*60)
    logger.info("Pipeline concluído com sucesso!")
    logger.info(f"Métricas finais: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, ROC-AUC={metrics.get('roc_auc', 'N/A')}")
    logger.info("="*60)

    return model, metrics


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Pipeline de treinamento do modelo')
    parser.add_argument('--data', type=str, default=None, help='Caminho para arquivo de dados')
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=['random_forest', 'gradient_boosting', 'logistic'],
                       help='Tipo de modelo')
    parser.add_argument('--no-compare', action='store_true', help='Não comparar modelos')

    args = parser.parse_args()

    model, metrics = run_pipeline(
        data_path=args.data,
        model_type=args.model,
        compare=not args.no_compare
    )
