"""
Módulo de pré-processamento de dados.
Contém funções para limpeza e preparação dos dados.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def load_data(file_path: str, sheet_name: str = 'PEDE2024') -> pd.DataFrame:
    """
    Carrega dados do arquivo Excel.

    Args:
        file_path: Caminho para o arquivo Excel
        sheet_name: Nome da planilha a ser carregada

    Returns:
        DataFrame com os dados carregados
    """
    logger.info(f"Carregando dados de {file_path}, planilha {sheet_name}")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    logger.info(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


def load_all_years(file_path: str) -> pd.DataFrame:
    """
    Carrega e combina dados de todos os anos (2022, 2023, 2024).

    Args:
        file_path: Caminho para o arquivo Excel

    Returns:
        DataFrame combinado com coluna 'ano' indicando a origem
    """
    xlsx = pd.ExcelFile(file_path)

    dfs = []
    for sheet in xlsx.sheet_names:
        df = pd.read_excel(xlsx, sheet_name=sheet)
        year = int(sheet.replace('PEDE', ''))
        df['ano_dados'] = year
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Dados combinados: {combined.shape[0]} linhas de {len(dfs)} anos")
    return combined


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Padroniza os nomes das colunas entre diferentes anos.

    Args:
        df: DataFrame com colunas originais

    Returns:
        DataFrame com nomes de colunas padronizados
    """
    df = df.copy()

    # Primeiro, remove colunas duplicadas (mantém a primeira)
    df = df.loc[:, ~df.columns.duplicated()]

    # Mapeamento de colunas - apenas renomeia se a coluna existir
    # Prioriza colunas mais recentes
    column_mapping = {}

    # Target
    if 'Defasagem' in df.columns:
        column_mapping['Defasagem'] = 'defasagem'
    elif 'Defas' in df.columns:
        column_mapping['Defas'] = 'defasagem'

    # Identificadores
    if 'RA' in df.columns:
        column_mapping['RA'] = 'ra'
    if 'Nome Anonimizado' in df.columns:
        column_mapping['Nome Anonimizado'] = 'nome'
    elif 'Nome' in df.columns:
        column_mapping['Nome'] = 'nome'

    # Demográficos
    if 'Gênero' in df.columns:
        column_mapping['Gênero'] = 'genero'
    if 'Idade' in df.columns:
        column_mapping['Idade'] = 'idade'
    elif 'Idade 22' in df.columns:
        column_mapping['Idade 22'] = 'idade'
    if 'Data de Nasc' in df.columns:
        column_mapping['Data de Nasc'] = 'data_nascimento'
    elif 'Ano nasc' in df.columns:
        column_mapping['Ano nasc'] = 'ano_nascimento'
    if 'Ano ingresso' in df.columns:
        column_mapping['Ano ingresso'] = 'ano_ingresso'

    # Educacionais
    if 'Fase' in df.columns:
        column_mapping['Fase'] = 'fase'
    if 'Fase Ideal' in df.columns:
        column_mapping['Fase Ideal'] = 'fase_ideal'
    elif 'Fase ideal' in df.columns:
        column_mapping['Fase ideal'] = 'fase_ideal'
    if 'Turma' in df.columns:
        column_mapping['Turma'] = 'turma'
    if 'Instituição de ensino' in df.columns:
        column_mapping['Instituição de ensino'] = 'instituicao'

    # Indicadores - prioriza versão mais recente
    for col in ['INDE 2024', 'INDE 2023', 'INDE 23', 'INDE 22']:
        if col in df.columns:
            column_mapping[col] = 'inde'
            break

    if 'IAA' in df.columns:
        column_mapping['IAA'] = 'iaa'
    if 'IEG' in df.columns:
        column_mapping['IEG'] = 'ieg'
    if 'IPS' in df.columns:
        column_mapping['IPS'] = 'ips'
    if 'IPP' in df.columns:
        column_mapping['IPP'] = 'ipp'
    if 'IDA' in df.columns:
        column_mapping['IDA'] = 'ida'
    if 'IPV' in df.columns:
        column_mapping['IPV'] = 'ipv'
    if 'IAN' in df.columns:
        column_mapping['IAN'] = 'ian'

    # Notas
    if 'Mat' in df.columns:
        column_mapping['Mat'] = 'mat'
    elif 'Matem' in df.columns:
        column_mapping['Matem'] = 'mat'
    if 'Por' in df.columns:
        column_mapping['Por'] = 'por'
    elif 'Portug' in df.columns:
        column_mapping['Portug'] = 'por'
    if 'Ing' in df.columns:
        column_mapping['Ing'] = 'ing'
    elif 'Inglês' in df.columns:
        column_mapping['Inglês'] = 'ing'

    # Pedra (classificação) - prioriza versão mais recente
    for col in ['Pedra 2024', 'Pedra 2023', 'Pedra 23', 'Pedra 22']:
        if col in df.columns:
            column_mapping[col] = 'pedra'
            break

    # Avaliações
    if 'Nº Av' in df.columns:
        column_mapping['Nº Av'] = 'num_avaliadores'

    # Status
    if 'Ativo/ Inativo' in df.columns:
        column_mapping['Ativo/ Inativo'] = 'status'

    df = df.rename(columns=column_mapping)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza limpeza dos dados.

    Args:
        df: DataFrame com dados brutos

    Returns:
        DataFrame limpo
    """
    df = df.copy()

    # Remove colunas duplicadas (ex: 'Ativo/ Inativo.1')
    cols_to_drop = [col for col in df.columns if col.endswith('.1')]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Remove colunas de identificação que não são úteis para ML
    id_cols = ['nome', 'Avaliador1', 'Avaliador2', 'Avaliador3', 'Avaliador4',
               'Avaliador5', 'Avaliador6', 'Rec Av1', 'Rec Av2', 'Rec Av3',
               'Rec Av4', 'Rec Psicologia', 'Cg', 'Cf', 'Ct']
    df = df.drop(columns=[c for c in id_cols if c in df.columns], errors='ignore')

    # Converte colunas numéricas
    numeric_cols = ['inde', 'iaa', 'ieg', 'ips', 'ipp', 'ida', 'ipv', 'ian',
                    'mat', 'por', 'ing', 'idade', 'ano_ingresso']

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    logger.info(f"Dados limpos: {df.shape[0]} linhas, {df.shape[1]} colunas")
    return df


def create_target(df: pd.DataFrame, binary: bool = True) -> pd.DataFrame:
    """
    Cria a variável target para o modelo.

    Args:
        df: DataFrame com coluna 'defasagem'
        binary: Se True, cria target binário (em risco vs não em risco)

    Returns:
        DataFrame com coluna 'target'
    """
    df = df.copy()

    if 'defasagem' not in df.columns:
        raise ValueError("Coluna 'defasagem' não encontrada no DataFrame")

    if binary:
        # Em risco: defasagem < 0 (está atrasado em relação à fase ideal)
        df['target'] = (df['defasagem'] < 0).astype(int)
    else:
        # Multi-classe: mantém a defasagem original
        df['target'] = df['defasagem']

    logger.info(f"Target criado. Distribuição: {df['target'].value_counts().to_dict()}")
    return df


def remove_leaky_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove features que vazam informação do target.

    Args:
        df: DataFrame com todas as features

    Returns:
        DataFrame sem features com vazamento
    """
    # IAN (Indicador de Adequação a Nível) é essencialmente o target
    # INDE é calculado a partir do IAN, portanto também contém vazamento
    # fase_ideal também contém informação do target
    leaky_cols = ['ian', 'inde', 'fase_ideal', 'defasagem']

    df = df.drop(columns=[c for c in leaky_cols if c in df.columns], errors='ignore')
    logger.info(f"Features com vazamento removidas: {leaky_cols}")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Retorna lista de colunas a serem usadas como features.

    Args:
        df: DataFrame

    Returns:
        Lista de nomes de colunas
    """
    exclude_cols = {'ra', 'target', 'ano_dados', 'turma', 'fase', 'status',
                    'data_nascimento', 'ano_nascimento', 'Escola'}

    feature_cols = [col for col in df.columns
                    if col not in exclude_cols
                    and not col.startswith('Pedra ')
                    and not col.startswith('INDE ')
                    and not col.startswith('Destaque')]

    return feature_cols


def split_features_target(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separa features e target.

    Args:
        df: DataFrame com features e target
        feature_cols: Lista de colunas para usar como features (opcional)

    Returns:
        Tupla (X, y) com features e target
    """
    if 'target' not in df.columns:
        raise ValueError("Coluna 'target' não encontrada. Execute create_target primeiro.")

    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    X = df[feature_cols].copy()
    y = df['target'].copy()

    return X, y
