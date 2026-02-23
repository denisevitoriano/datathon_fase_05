# Dockerfile para API de Predição - Passos Mágicos
FROM python:3.11-slim

# Metadados
LABEL maintainer="Passos Mágicos ML Team"
LABEL description="API para predição de risco de defasagem escolar"
LABEL version="1.0.0"

# Variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Diretório de trabalho
WORKDIR /app

# Instala dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements e instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia código fonte
COPY src/ ./src/
COPY app/ ./app/

# Copia modelo e artefatos (se existirem)
COPY app/model/*.joblib ./app/model/
COPY app/model/*.json ./app/model/

# Cria diretório de logs
RUN mkdir -p logs

# Expõe porta da API
EXPOSE 8000

# Comando de saúde
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Comando padrão
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
