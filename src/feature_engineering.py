"""
Módulo de engenharia de features.
Contém funções para criação e transformação de features.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Dict, Any
import logging
import joblib

logger = logging.getLogger(__name__)


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria features derivadas a partir das existentes.

    Args:
        df: DataFrame com features originais

    Returns:
        DataFrame com features adicionais
    """
    df = df.copy()

    # Média das notas (se disponíveis)
    grade_cols = ['mat', 'por', 'ing']
    available_grades = [c for c in grade_cols if c in df.columns]
    if available_grades:
        df['media_notas'] = df[available_grades].mean(axis=1)
        df['min_nota'] = df[available_grades].min(axis=1)
        df['max_nota'] = df[available_grades].max(axis=1)
        df['std_notas'] = df[available_grades].std(axis=1)

    # Anos no programa (se ano_ingresso disponível)
    if 'ano_ingresso' in df.columns and 'ano_dados' in df.columns:
        df['anos_no_programa'] = df['ano_dados'] - df['ano_ingresso']
    elif 'ano_ingresso' in df.columns:
        # Assume dados de 2024 se não houver ano_dados
        df['anos_no_programa'] = 2024 - df['ano_ingresso']

    # Indicador composto de desempenho
    indicator_cols = ['inde', 'iaa', 'ieg', 'ips', 'ida', 'ipv']
    available_indicators = [c for c in indicator_cols if c in df.columns]
    if available_indicators:
        df['media_indicadores'] = df[available_indicators].mean(axis=1)

    # Razão entre indicadores (se disponíveis)
    if 'ida' in df.columns and 'ieg' in df.columns:
        df['ratio_ida_ieg'] = df['ida'] / (df['ieg'] + 0.001)

    if 'ipv' in df.columns and 'ips' in df.columns:
        df['ratio_ipv_ips'] = df['ipv'] / (df['ips'] + 0.001)

    logger.info(f"Features derivadas criadas. Total de colunas: {len(df.columns)}")
    return df


def get_numeric_features(df: pd.DataFrame) -> List[str]:
    """
    Identifica colunas numéricas para o modelo.

    Args:
        df: DataFrame

    Returns:
        Lista de nomes de colunas numéricas
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove colunas que não devem ser features
    exclude = {'target', 'defasagem', 'ian', 'ano_dados'}
    numeric_cols = [c for c in numeric_cols if c not in exclude]

    return numeric_cols


def get_categorical_features(df: pd.DataFrame) -> List[str]:
    """
    Identifica colunas categóricas para o modelo.

    Args:
        df: DataFrame

    Returns:
        Lista de nomes de colunas categóricas
    """
    # Colunas categóricas úteis
    potential_cat = ['genero', 'instituicao', 'pedra']

    cat_cols = [c for c in potential_cat if c in df.columns]
    return cat_cols


def create_preprocessing_pipeline(
    numeric_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """
    Cria pipeline de pré-processamento com transformadores.

    Args:
        numeric_features: Lista de features numéricas
        categorical_features: Lista de features categóricas

    Returns:
        ColumnTransformer configurado
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    return preprocessor


def fit_preprocessor(
    X: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str]
) -> Tuple[ColumnTransformer, np.ndarray]:
    """
    Treina o preprocessador e transforma os dados.

    Args:
        X: DataFrame com features
        numeric_features: Lista de features numéricas
        categorical_features: Lista de features categóricas

    Returns:
        Tupla (preprocessor, X_transformed)
    """
    preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)
    X_transformed = preprocessor.fit_transform(X)

    logger.info(f"Preprocessador treinado. Shape transformado: {X_transformed.shape}")
    return preprocessor, X_transformed


def transform_data(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer
) -> np.ndarray:
    """
    Aplica transformações usando preprocessador já treinado.

    Args:
        X: DataFrame com features
        preprocessor: Preprocessador já treinado

    Returns:
        Array numpy com dados transformados
    """
    return preprocessor.transform(X)


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Obtém nomes das features após transformação.

    Args:
        preprocessor: Preprocessador já treinado

    Returns:
        Lista de nomes das features
    """
    try:
        return preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        # Fallback para versões antigas do sklearn
        return [f"feature_{i}" for i in range(preprocessor.n_features_in_)]


def save_preprocessor(preprocessor: ColumnTransformer, path: str) -> None:
    """
    Salva o preprocessador em arquivo.

    Args:
        preprocessor: Preprocessador treinado
        path: Caminho para salvar
    """
    joblib.dump(preprocessor, path)
    logger.info(f"Preprocessador salvo em {path}")


def load_preprocessor(path: str) -> ColumnTransformer:
    """
    Carrega preprocessador de arquivo.

    Args:
        path: Caminho do arquivo

    Returns:
        Preprocessador carregado
    """
    preprocessor = joblib.load(path)
    logger.info(f"Preprocessador carregado de {path}")
    return preprocessor


def prepare_features_for_training(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Prepara features para treinamento.

    Args:
        df: DataFrame com dados limpos

    Returns:
        Tupla (df_with_features, numeric_features, categorical_features)
    """
    # Cria features derivadas
    df = create_derived_features(df)

    # Identifica tipos de features
    numeric_features = get_numeric_features(df)
    categorical_features = get_categorical_features(df)

    logger.info(f"Features numéricas: {len(numeric_features)}")
    logger.info(f"Features categóricas: {len(categorical_features)}")

    return df, numeric_features, categorical_features
