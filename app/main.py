"""
API FastAPI para predição de risco de defasagem escolar.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator

from app.routes import router
from app.monitoring_routes import router as monitoring_router
from app.model.predictor import load_model_artifacts

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager para carregar modelo na inicialização."""
    logger.info("Carregando modelo e preprocessador...")
    try:
        load_model_artifacts()
        logger.info("Modelo carregado com sucesso!")
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        raise
    yield
    logger.info("Encerrando aplicação...")


app = FastAPI(
    title="Passos Mágicos - API de Predição",
    description="""
    API para predição de risco de defasagem escolar de estudantes.

    ## Funcionalidades

    * **Predição individual**: Recebe dados de um estudante e retorna probabilidade de risco
    * **Predição em lote**: Processa múltiplos estudantes de uma vez
    * **Health check**: Verifica status da API e do modelo

    ## Sobre o Modelo

    O modelo foi treinado com dados da Associação Passos Mágicos para identificar
    estudantes em risco de defasagem escolar, permitindo intervenções precoces.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rotas
app.include_router(router)
app.include_router(monitoring_router)


@app.get("/", tags=["Default"])
async def root():
    """Endpoint raiz com informações da API."""
    return {
        "name": "Passos Mágicos - API de Predição",
        "version": "1.0.0",
        "description": "API para predição de risco de defasagem escolar",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "health": "/health",
            "model_info": "/model/info",
            "monitoring_dashboard": "/monitoring/dashboard",
            "drift_status": "/monitoring/drift-status"
        },
        "timestamp": datetime.now().isoformat()
    }

Instrumentator().instrument(app).expose(app,
                                        endpoint="/metrics",
                                        include_in_schema=False)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
