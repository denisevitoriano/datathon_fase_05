"""
Schemas Pydantic para validação de dados da API.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
from enum import Enum


class GenderEnum(str, Enum):
    masculino = "Masculino"
    feminino = "Feminino"


class InstitutionEnum(str, Enum):
    publica = "Pública"
    privada = "Privada"


class PedraEnum(str, Enum):
    ametista = "Ametista"
    topazio = "Topázio"
    quartzo = "Quartzo"
    agata = "Agata"


class StudentInput(BaseModel):
    """Schema para entrada de dados de um estudante."""

    # Indicadores principais (escala 0-10)
    inde: Optional[float] = Field(None, ge=0, le=10, description="Índice de Desenvolvimento Educacional")
    iaa: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Auto-Avaliação")
    ieg: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Engajamento")
    ips: Optional[float] = Field(None, ge=0, le=10, description="Indicador Psicossocial")
    ipp: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Ponto de Virada (Percepção)")
    ida: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Desempenho Acadêmico")
    ipv: Optional[float] = Field(None, ge=0, le=10, description="Indicador de Ponto de Virada")

    # Notas (escala 0-10)
    mat: Optional[float] = Field(None, ge=0, le=10, description="Nota em Matemática")
    por: Optional[float] = Field(None, ge=0, le=10, description="Nota em Português")
    ing: Optional[float] = Field(None, ge=0, le=10, description="Nota em Inglês")

    # Dados demográficos
    idade: Optional[int] = Field(None, ge=5, le=30, description="Idade do estudante")
    ano_ingresso: Optional[int] = Field(None, ge=2000, le=2030, description="Ano de ingresso no programa")
    num_avaliacoes: Optional[int] = Field(None, ge=0, le=10, description="Número de avaliações")

    # Dados categóricos
    genero: Optional[str] = Field(None, description="Gênero (Masculino/Feminino)")
    instituicao: Optional[str] = Field(None, description="Tipo de instituição (Pública/Privada)")
    pedra: Optional[str] = Field(None, description="Classificação Pedra (Ametista/Topázio/Quartzo/Agata)")

    class Config:
        json_schema_extra = {
            "example": {
                "inde": 7.5,
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
                "num_avaliacoes": 3,
                "genero": "Masculino",
                "instituicao": "Pública",
                "pedra": "Topázio"
            }
        }


class PredictionOutput(BaseModel):
    """Schema para saída de predição."""

    at_risk: bool = Field(..., description="Se o estudante está em risco de defasagem")
    risk_probability: float = Field(..., ge=0, le=1, description="Probabilidade de risco (0-1)")
    risk_level: str = Field(..., description="Nível de risco (Baixo/Médio/Alto)")
    confidence: float = Field(..., ge=0, le=1, description="Confiança da predição")

    class Config:
        json_schema_extra = {
            "example": {
                "at_risk": True,
                "risk_probability": 0.73,
                "risk_level": "Alto",
                "confidence": 0.85
            }
        }


class BatchInput(BaseModel):
    """Schema para entrada em lote."""

    students: List[StudentInput] = Field(..., min_length=1, max_length=1000,
                                         description="Lista de estudantes para predição")


class BatchPredictionOutput(BaseModel):
    """Schema para saída de predição em lote."""

    predictions: List[PredictionOutput]
    total_processed: int
    at_risk_count: int
    at_risk_percentage: float


class HealthResponse(BaseModel):
    """Schema para resposta de health check."""

    status: str
    model_loaded: bool
    model_type: Optional[str]
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Schema para informações do modelo."""

    model_type: str
    metrics: dict
    features: dict
    training_info: dict
