"""
Rotas de monitoramento da API.
Endpoints para drift detection e métricas.
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Optional
import pandas as pd
import json
from pathlib import Path

router = APIRouter(prefix="/monitoring", tags=["Monitoramento"])

# Armazenamento em memória para demo
_prediction_history = []
_drift_reports = []


@router.get("/dashboard")
async def get_dashboard():
    """
    Retorna dados para o dashboard de monitoramento.

    Inclui:
    - Sumário de predições
    - Histórico de drift
    - Métricas de performance
    """
    total_predictions = len(_prediction_history)
    at_risk = sum(1 for p in _prediction_history if p.get('prediction') == 1)

    return {
        "timestamp": datetime.now().isoformat(),
        "prediction_summary": {
            "total": total_predictions,
            "at_risk": at_risk,
            "not_at_risk": total_predictions - at_risk,
            "at_risk_rate": at_risk / total_predictions if total_predictions > 0 else 0
        },
        "drift_reports_count": len(_drift_reports),
        "last_drift_check": _drift_reports[-1] if _drift_reports else None
    }


@router.post("/log-prediction")
async def log_prediction(data: dict):
    """
    Registra uma predição para monitoramento.

    Args:
        data: Dados da predição (input, prediction, probability)
    """
    _prediction_history.append({
        "timestamp": datetime.now().isoformat(),
        **data
    })

    # Mantém apenas últimas 10000 predições em memória
    if len(_prediction_history) > 10000:
        _prediction_history.pop(0)

    return {"status": "logged", "total_predictions": len(_prediction_history)}


@router.get("/prediction-history")
async def get_prediction_history(limit: int = 100):
    """
    Retorna histórico de predições.

    Args:
        limit: Número máximo de registros a retornar
    """
    return {
        "total": len(_prediction_history),
        "limit": limit,
        "predictions": _prediction_history[-limit:]
    }


@router.get("/drift-status")
async def get_drift_status():
    """
    Retorna status atual de drift.
    """
    if not _drift_reports:
        return {
            "status": "no_reports",
            "message": "Nenhum relatório de drift disponível"
        }

    latest = _drift_reports[-1]

    return {
        "status": "drift_detected" if latest.get("drift_detected") else "no_drift",
        "last_check": latest.get("timestamp"),
        "features_with_drift": latest.get("drift_count", 0),
        "total_features": latest.get("total_features", 0)
    }


@router.post("/check-drift")
async def check_drift(data: dict):
    """
    Executa verificação de drift com dados fornecidos.

    Args:
        data: Dicionário com 'current_data' (lista de registros)
    """
    try:
        from src.monitoring import DriftDetector

        # Carrega dados de referência
        reference_path = Path("data/processed/reference_data.csv")

        if not reference_path.exists():
            return {
                "status": "error",
                "message": "Dados de referência não encontrados"
            }

        reference_data = pd.read_csv(reference_path)
        current_data = pd.DataFrame(data.get("current_data", []))

        if current_data.empty:
            return {
                "status": "error",
                "message": "Dados atuais não fornecidos"
            }

        # Detecta drift
        feature_columns = reference_data.select_dtypes(include=['number']).columns.tolist()
        detector = DriftDetector(reference_data, feature_columns)
        drift_result = detector.detect_drift(current_data)

        _drift_reports.append(drift_result)

        return drift_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics-summary")
async def get_metrics_summary():
    """
    Retorna sumário de métricas do modelo.
    """
    try:
        # Tenta carregar métricas salvas
        metrics_path = Path("logs/model_card.json")

        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                model_card = json.load(f)
                return {
                    "model_type": model_card.get("model_name"),
                    "metrics": model_card.get("metrics", {}),
                    "training_info": model_card.get("training_data", {})
                }

        return {"status": "no_metrics", "message": "Métricas não disponíveis"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
