"""
Rotas da API.
Define endpoints para predição e monitoramento.
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
import logging
import time

from app.metrics import (
    PREDICTIONS_TOTAL, PREDICTION_PROBABILITY,
    PREDICTION_LATENCY, BATCH_SIZE, AT_RISK_RATE
)
from app.schemas import (
    StudentInput, PredictionOutput, BatchInput, BatchPredictionOutput,
    HealthResponse, ModelInfoResponse
)
from app.model.predictor import (
    predict, predict_batch, get_risk_level, get_confidence,
    is_model_loaded, get_model_metadata, get_features_config
)

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/predict", response_model=PredictionOutput, tags=["Predição"])
async def predict_risk(student: StudentInput):
    """
    Prediz risco de defasagem escolar para um estudante.

    - **inde**: Índice de Desenvolvimento Educacional (0-10)
    - **iaa**: Indicador de Auto-Avaliação (0-10)
    - **ieg**: Indicador de Engajamento (0-10)
    - **ips**: Indicador Psicossocial (0-10)
    - **ida**: Indicador de Desempenho Acadêmico (0-10)
    - **mat, por, ing**: Notas (0-10)
    - **idade**: Idade do estudante
    - **genero**: Masculino/Feminino
    - **instituicao**: Pública/Privada
    """
    try:
        # Converte para dicionário
        data = student.model_dump()

        # Faz predição com medição de tempo
        start = time.perf_counter()
        prediction, probability = predict(data)
        PREDICTION_LATENCY.observe(time.perf_counter() - start)

        # Atualiza métricas Prometheus
        risk_label = "at_risk" if prediction == 1 else "not_at_risk"
        PREDICTIONS_TOTAL.labels(risk_level=risk_label).inc()
        PREDICTION_PROBABILITY.observe(probability)
        AT_RISK_RATE.set(probability if prediction == 1 else 0)

        # Monta resposta
        return PredictionOutput(
            at_risk=bool(prediction),
            risk_probability=round(probability, 4),
            risk_level=get_risk_level(probability),
            confidence=round(get_confidence(probability), 4)
        )

    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar predição: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionOutput, tags=["Predição"])
async def predict_risk_batch(batch: BatchInput):
    """
    Prediz risco de defasagem escolar para múltiplos estudantes.

    Aceita uma lista de até 1000 estudantes e retorna predições para todos.
    """
    try:
        # Converte para lista de dicionários
        data_list = [s.model_dump() for s in batch.students]

        # Faz predições em lote com medição de tempo
        start = time.perf_counter()
        results = predict_batch(data_list)
        elapsed = time.perf_counter() - start
        PREDICTION_LATENCY.observe(elapsed)
        BATCH_SIZE.observe(len(data_list))

        # Monta respostas
        predictions = []
        at_risk_count = 0

        for prediction, probability in results:
            risk_label = "at_risk" if prediction == 1 else "not_at_risk"
            PREDICTIONS_TOTAL.labels(risk_level=risk_label).inc()
            PREDICTION_PROBABILITY.observe(probability)

            if prediction == 1:
                at_risk_count += 1

            predictions.append(PredictionOutput(
                at_risk=bool(prediction),
                risk_probability=round(probability, 4),
                risk_level=get_risk_level(probability),
                confidence=round(get_confidence(probability), 4)
            ))

        total = len(predictions)
        AT_RISK_RATE.set(at_risk_count / total * 100 if total > 0 else 0)

        return BatchPredictionOutput(
            predictions=predictions,
            total_processed=total,
            at_risk_count=at_risk_count,
            at_risk_percentage=round(at_risk_count / total * 100, 2) if total > 0 else 0
        )

    except Exception as e:
        logger.error(f"Erro na predição em lote: {e}")
        raise HTTPException(status_code=500, detail=f"Erro ao processar predições: {str(e)}")


@router.get("/health", response_model=HealthResponse, tags=["Monitoramento"])
async def health_check():
    """
    Verifica status da API e do modelo.

    Retorna informações sobre:
    - Status da API
    - Se o modelo está carregado
    - Tipo do modelo
    """
    metadata = get_model_metadata() if is_model_loaded() else {}

    return HealthResponse(
        status="healthy" if is_model_loaded() else "unhealthy",
        model_loaded=is_model_loaded(),
        model_type=metadata.get('model_type'),
        timestamp=datetime.now().isoformat()
    )


@router.get("/model/info", response_model=ModelInfoResponse, tags=["Monitoramento"])
async def model_info():
    """
    Retorna informações detalhadas sobre o modelo.

    Inclui:
    - Tipo do modelo
    - Métricas de performance
    - Features utilizadas
    - Informações de treinamento
    """
    if not is_model_loaded():
        raise HTTPException(status_code=503, detail="Modelo não carregado")

    metadata = get_model_metadata()
    features = get_features_config()

    return ModelInfoResponse(
        model_type=metadata.get('model_type', 'unknown'),
        metrics=metadata.get('metrics', {}),
        features=features,
        training_info={
            'feature_count': len(features.get('all_features', [])),
            'numeric_features': len(features.get('numeric_features', [])),
            'categorical_features': len(features.get('categorical_features', []))
        }
    )


@router.get("/features", tags=["Informações"])
async def list_features():
    """
    Lista as features esperadas pelo modelo.

    Retorna informações sobre quais dados devem ser enviados
    para fazer uma predição.
    """
    features = get_features_config()

    return {
        "numeric_features": {
            "description": "Features numéricas (valores contínuos)",
            "features": features.get('numeric_features', [])
        },
        "categorical_features": {
            "description": "Features categóricas",
            "features": features.get('categorical_features', [])
        },
        "total": len(features.get('all_features', []))
    }
