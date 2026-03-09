# Passos Mágicos - Modelo de Predição de Defasagem Escolar

Sistema de Machine Learning para identificação de estudantes em risco de defasagem escolar, desenvolvido para a Associação Passos Mágicos.

## 1. Visão Geral do Projeto

### Objetivo
Desenvolver um modelo preditivo capaz de estimar o **risco de defasagem escolar** de cada estudante, permitindo intervenções precoces e direcionadas.

### Solução Proposta
Pipeline completa de Machine Learning incluindo:
- Pré-processamento e limpeza de dados
- Engenharia de features
- Treinamento e validação de modelos
- API para predições em tempo real
- Monitoramento contínuo e detecção de drift
- Dashboard de monitoramento (Grafana + Prometheus)
- Testes automatizados com cobertura
- Containerização com Docker e Docker Compose

### Stack Tecnológica

| Componente | Tecnologia |
|------------|------------|
| Linguagem | Python 3.10+ |
| Gerenciador de Pacotes | uv |
| Frameworks ML | scikit-learn, pandas, numpy |
| API | FastAPI |
| Serialização | joblib |
| Testes | pytest (207 testes, 96% cobertura) |
| Empacotamento | Docker |
| Monitoramento | Prometheus + Grafana + drift detection |

## 2. Estrutura do Projeto

```
datathon_fase_05/
├── app/                          # API FastAPI
│   ├── main.py                   # Aplicação principal
│   ├── metrics.py                # Métricas Prometheus do modelo
│   ├── monitoring_routes.py      # Endpoints de monitoramento
│   ├── routes.py                 # Endpoints de predição
│   ├── schemas.py                # Schemas Pydantic
│   └── model/                    # Artefatos do modelo
│       ├── features_config.json  # Configuração de features
│       ├── model.joblib          # Modelo treinado
│       ├── predictor.py          # Lógica de predição
│       └── preprocessor.joblib   # Preprocessador
│
├── dashboard/
│   └── dashboard-datathon-passos-magicos.json  # Dashboard Grafana
│
├── grafana/
│   └── provisioning/
│       ├── datasources/
│       │   └── datasource.yml    # Configuração automática do Prometheus
│       └── dashboards/
│           └── dashboard.yml     # Provisionamento automático do dashboard
│
├── data/
│   ├── processed/                 # Dados pré-processados
│   ├── raw/                       # Dados brutos
│   └── train/                     # Dados de treino (features + target)
│
├── src/                          # Pipeline de ML
│   ├── evaluate.py               # Avaliação e métricas
│   ├── feature_engineering.py    # Criação de features
│   ├── monitoring.py             # Detecção de drift
│   ├── pipeline.py               # Pipeline principal
│   ├── preprocessing.py          # Limpeza e preparação de dados
│   ├── train.py                  # Treinamento do modelo
│   └── utils.py                  # Funções auxiliares
│
├── tests/                        # Testes unitários
│   ├── test_api.py               # Testes da API
│   ├── test_evaluate.py          # Testes de avaliação
│   ├── test_feature_engineering.py  # Testes de engenharia de features
│   ├── test_model.py             # Testes do modelo
│   ├── test_monitoring.py        # Testes de monitoramento
│   ├── test_monitoring_routes.py # Testes das rotas de monitoramento
│   ├── test_pipeline.py          # Testes do pipeline
│   ├── test_predictor.py         # Testes do predictor
│   ├── test_preprocessing.py     # Testes de pré-processamento
│   └── test_utils.py             # Testes de utilitários
│
├── docker-compose.yml            # Orquestração de containers
├── Dockerfile                    # Imagem Docker da API
├── prometheus.yaml               # Configuração do Prometheus
├── pyproject.toml                # Configuração do projeto e dependências
├── pytest.ini                    # Configuração dos testes
├── README.md                     # Documentação do projeto
├── requirements.txt              # Dependências exportadas (pip)
└── uv.lock                       # Lockfile do uv
```

## 3. Instruções de Deploy

### Pré-requisitos
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes)
- Docker

### Instalação do uv

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Ou via pip (se necessário)
pip install uv
```

### Instalação Local

```bash
# Clone o repositório
git clone <repository-url>
cd datathon_fase_05

# Instale as dependências (cria .venv automaticamente)
uv sync --all-extras
```

### Treinamento do Modelo

```bash
# Execute a pipeline de treinamento
uv run python -m src.pipeline

# Ou especifique opções
uv run python -m src.pipeline --model random_forest --no-compare
```

### Executar API Localmente

```bash
# Inicie o servidor
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Acesse a documentação interativa
http://localhost:8000/docs
```

### Deploy com Docker

#### Pré-requisito
Certifique-se de que o **Docker Desktop** está instalado e **rodando** antes de executar os comandos abaixo.

#### Opção 1: Docker Compose (recomendado)

O `docker-compose` sobe toda a stack de uma vez: API, Prometheus e Grafana, já com rede, volumes e restart automático configurados.

```bash
# Build e inicialização de todos os serviços
docker-compose up --build

# Para rodar em segundo plano (modo detached)
docker-compose up --build -d
```

Após a inicialização, os seguintes serviços estarão disponíveis:

| Serviço | URL | Descrição |
|---------|-----|-----------|
| API FastAPI | http://localhost:8000 | API de predição (docs em `/docs`) |
| Prometheus | http://localhost:9090 | Coleta e consulta de métricas |
| Grafana | http://localhost:3000 | Dashboards de monitoramento (login: `admin` / `admin`) |

**Comandos úteis do docker-compose:**

```bash
# Verificar status dos containers
docker-compose ps

# Ver logs de todos os serviços
docker-compose logs -f

# Ver logs de um serviço específico
docker-compose logs -f api

# Parar todos os serviços
docker-compose down

# Parar e remover volumes (apaga dados do Prometheus/Grafana)
docker-compose down -v
```

#### Opção 2: Docker standalone (apenas a API)

Se você precisa rodar **somente a API**, sem Prometheus e Grafana:

```bash
# Build da imagem
docker build -t passos-magicos-api .

# Execute o container
docker run -p 8000:8000 passos-magicos-api
```

### Executar Testes

```bash
# Execute todos os testes com cobertura
uv run pytest

# Ou execute testes específicos
uv run pytest tests/test_preprocessing.py -v

# Gere relatório de cobertura HTML
uv run pytest --cov-report=html
```

### Checklist de Validação

- [ ] Ambiente instalado com `uv sync --all-extras`
- [ ] Modelo treinado com `uv run python -m src.pipeline`
- [ ] Artefatos salvos em `app/model/` (model.joblib, preprocessor.joblib, features_config.json)
- [ ] API inicia sem erros
- [ ] Health check retorna `status: healthy`
- [ ] Predição individual funciona (`/predict`)
- [ ] Predição em lote funciona (`/predict/batch`)
- [ ] Swagger UI acessível em `/docs`
- [ ] Testes passando com 96% de cobertura
- [ ] Docker Compose sobe API, Prometheus e Grafana
- [ ] Dashboard Grafana carregado automaticamente (provisionamento)

## 4. Exemplos de Chamadas à API

### Informações do Modelo

```bash
curl http://localhost:8000/model/info
```

**Resposta:**
```json
{
  "model_type": "random_forest",
  "metrics": {
    "accuracy": 0.8147,
    "precision": 0.7759,
    "recall": 0.8411,
    "f1": 0.8072,
    "roc_auc": 0.8577,
    "cv_results": {
      "accuracy_mean": 0.7391,
      "accuracy_std": 0.0188,
      "precision_mean": 0.6855,
      "precision_std": 0.0273,
      "recall_mean": 0.8104,
      "recall_std": 0.0424,
      "f1_mean": 0.7416,
      "f1_std": 0.0161,
      "roc_auc_mean": 0.8137,
      "roc_auc_std": 0.0187
    }
  },
  "features": {
    "numeric_features": [
      "idade", "iaa", "ieg", "ips", "ipp", "ida", "mat", "por", "ing", "ipv",
      "media_notas", "min_nota", "max_nota", "std_notas",
      "anos_no_programa", "media_indicadores", "ratio_ida_ieg", "ratio_ipv_ips"
    ],
    "categorical_features": ["genero", "instituicao"],
    "all_features": [
      "idade", "iaa", "ieg", "ips", "ipp", "ida", "mat", "por", "ing", "ipv",
      "media_notas", "min_nota", "max_nota", "std_notas",
      "anos_no_programa", "media_indicadores", "ratio_ida_ieg", "ratio_ipv_ips",
      "genero", "instituicao"
    ]
  },
  "training_info": {
    "feature_count": 20,
    "numeric_features": 18,
    "categorical_features": 2
  }
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

**Resposta:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "random_forest",
  "timestamp": "2026-03-08T23:33:26.263013"
}
```

### Predição Individual

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "genero": "Masculino",
    "instituicao": "Pública"
  }'
```

**Resposta:**
```json
{
  "at_risk": true,
  "risk_probability": 0.5047,
  "risk_level": "Médio",
  "confidence": 0.0095
}
```

### Predição em Lote

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "students": [
      {"iaa": 8.0, "ieg": 7.0, "ips": 6.5, "idade": 12},
      {"iaa": 5.5, "ieg": 6.0, "ips": 5.0, "idade": 14}
    ]
  }'
```

**Resposta:**
```json
{
  "predictions": [
    {"at_risk": false, "risk_probability": 0.4466, "risk_level": "Médio", "confidence": 0.1068},
    {"at_risk": false, "risk_probability": 0.4329, "risk_level": "Médio", "confidence": 0.1342}
  ],
  "total_processed": 2,
  "at_risk_count": 0,
  "at_risk_percentage": 0.0
}
```

### Python

```python
import requests

# Predição individual
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "iaa": 8.0,
        "ieg": 7.0,
        "ips": 6.5,
        "idade": 12,
        "genero": "Masculino"
    }
)
print(response.json())
```

**Resposta:**
```json
{
  "at_risk": false,
  "risk_probability": 0.4441,
  "risk_level": "Médio",
  "confidence": 0.1118
}
```

> **Dica:** Para executar sem instalar dependências manualmente, use o `uv`:
> ```bash
> uv run --with requests python -c "import requests; r = requests.post('http://localhost:8000/predict', json={'iaa': 8.0, 'ieg': 7.0, 'ips': 6.5, 'idade': 12, 'genero': 'Masculino'}); print(r.json())"
> ```

## 5. Etapas do Pipeline de Machine Learning

### 5.1 Pré-processamento dos Dados
- Carregamento dos dados do Excel (planilha PEDE2024)
- Padronização de nomes de colunas (mapeamento de diferentes versões/anos)
- Remoção de colunas duplicadas e de identificação (nome, avaliadores, etc.)
- Conversão de tipos de dados numéricos
- Criação do target binário a partir da coluna `defasagem`
- Remoção de features com vazamento de dados (IAN, INDE, fase_ideal, defasagem)

### 5.2 Engenharia de Features
- **Features numéricas (18)**: idade, iaa, ieg, ips, ipp, ida, ipv, mat, por, ing + features derivadas
- **Features categóricas (2)**: genero, instituicao
- **Features derivadas**:
  - `media_notas`: média das notas (mat, por, ing)
  - `min_nota`, `max_nota`, `std_notas`: estatísticas das notas
  - `anos_no_programa`: calculado a partir do ano de ingresso
  - `media_indicadores`: média dos indicadores (iaa, ieg, ips, ida, ipv)
  - `ratio_ida_ieg`: razão entre IDA e IEG
  - `ratio_ipv_ips`: razão entre IPV e IPS
- **Pré-processamento das features**:
  - Numéricas: imputação pela mediana + padronização (StandardScaler)
  - Categóricas: imputação por valor constante + OneHotEncoder

### 5.3 Variável Target
A variável `defasagem` representa a diferença entre a fase atual do estudante e a fase ideal baseada na idade:

| Defasagem | Interpretação |
|-----------|---------------|
| -2 ou menos | Alto risco - 2+ anos atrasado |
| -1 | Risco moderado - 1 ano atrasado |
| 0 | Adequado - na fase correta |
| +1 ou mais | Adiantado |

Para o modelo, a defasagem é convertida em **classificação binária**:
- **Em risco (1)**: defasagem < 0 (atrasado em relação à fase ideal)
- **Não risco (0)**: defasagem >= 0

### 5.4 Treinamento e Validação
- **Divisão treino/teste**: realizada antes do pré-processamento para evitar vazamento de dados
- **Modelos avaliados**:
  - Random Forest (`class_weight='balanced'`)
  - Gradient Boosting
- **Validação**: Cross-validation estratificada (5 folds) no conjunto de treino

### 5.5 Seleção de Modelo
- Comparação automática dos modelos avaliados
- Seleção baseada em F1-Score (melhor equilíbrio entre precisão e recall)
- Geração de ranking de importância das features

### 5.6 Métricas de Avaliação
- **Accuracy**: Proporção de predições corretas
- **Precision**: Proporção de verdadeiros positivos entre preditos positivos
- **Recall**: Proporção de verdadeiros positivos identificados
- **F1-Score**: Média harmônica de precision e recall
- **ROC-AUC**: Área sob a curva ROC

### 5.7 Classificação de Risco (Saída do Modelo)

| Probabilidade | Nível de Risco | Recomendação |
|---------------|----------------|--------------|
| < 0.3 | Baixo | Acompanhamento regular |
| 0.3 - 0.7 | Médio | Atenção especial |
| > 0.7 | Alto | Intervenção prioritária |

## 6. Monitoramento do Modelo

A API expõe endpoints dedicados ao monitoramento do modelo, incluindo histórico de predições, detecção de drift e métricas de performance.

### Endpoints de Monitoramento

| Endpoint | Método | Descrição |
|----------|--------|-----------|
| `/monitoring/dashboard` | GET | Dashboard com sumário de predições e status de drift |
| `/monitoring/drift-status` | GET | Status atual de drift |
| `/monitoring/check-drift` | POST | Executa verificação de drift com dados fornecidos |
| `/monitoring/prediction-history` | GET | Histórico de predições (parâmetro `limit`, default: 100) |
| `/monitoring/log-prediction` | POST | Registra uma predição para monitoramento |
| `/monitoring/metrics-summary` | GET | Métricas do modelo carregadas do model card |

### Detecção de Drift
O sistema detecta mudanças na distribuição dos dados de entrada em relação aos dados de treino usando dois métodos estatísticos:

- **Kolmogorov-Smirnov Test**: Compara distribuições entre dados de referência e dados atuais (threshold: p-value < 0.05)
- **Population Stability Index (PSI)**: Quantifica a magnitude do drift usando bins percentuais (threshold: PSI > 0.2)

Drift é reportado quando **qualquer um** dos thresholds é ultrapassado em pelo menos uma feature.

## 7. Monitoramento da Aplicação

A aplicação utiliza **Prometheus** e **Grafana** para monitoramento em tempo real, com métricas expostas automaticamente no endpoint `/metrics` via `prometheus-fastapi-instrumentator` e métricas customizadas do modelo via `prometheus-client`.

### Arquitetura

```
API FastAPI ──(/metrics)──> Prometheus ──(datasource)──> Grafana
   :8000                      :9090                       :3000
```

- **Prometheus** coleta métricas da API a cada 15 segundos (configurado em `prometheus.yaml`)
- **Grafana** consome os dados do Prometheus e exibe nos dashboards
- O datasource e o dashboard são **provisionados automaticamente** ao subir os containers (via `grafana/provisioning/`)

### Dashboard

O dashboard "Datathon - Passos Mágicos" é carregado automaticamente ao subir o Grafana via `docker-compose`. Basta acessar http://localhost:3000 (login: `admin` / `admin`) e o dashboard já estará disponível.

O dashboard é dividido em duas seções:

#### Seção 1: Latência, Tráfego e Taxa de Erros (Four Golden Signals)

Construída seguindo os "Four Golden Signals" do livro de SRE do Google:

| Painel | Tipo | Descrição |
|--------|------|-----------|
| Painel de Latência | Time series | P99, P95 e P50 do tempo de resposta das requisições HTTP |
| Painel de Tráfego | Time series | Requests/s agrupados por código de status HTTP |
| Painel de Taxa de Erros | Gauge | Percentual de requisições com status 4xx/5xx |
| Saturação de CPU | Gauge | Uso de CPU do processo da API |
| Saturação de Memória | Gauge | Uso de memória do processo em MB |

#### Seção 2: Métricas do Modelo

Métricas específicas do modelo de ML, instrumentadas via `prometheus-client`:

| Painel | Tipo | Descrição |
|--------|------|-----------|
| Predições por Classificação | Time series | Volume de predições "Em Risco" vs "Sem Risco" ao longo do tempo |
| Latência de Inferência | Time series | P99 e P50 do tempo de inferência do modelo (sem overhead HTTP) |
| Taxa de Estudantes em Risco | Gauge | Percentual de estudantes classificados em risco |
| Total de Predições | Stat | Contador acumulado de predições realizadas |
| Status de Data Drift | Stat | Indica se drift foi detectado (verde = sem drift, vermelho = drift) |
| Features com Drift | Stat | Número de features com drift vs total monitoradas |
| Distribuição de Probabilidade de Risco | Bar chart | Histograma das probabilidades retornadas pelo modelo |

### Métricas Prometheus Expostas

#### Métricas de infraestrutura (via `prometheus-fastapi-instrumentator`)

| Métrica | Tipo | Descrição |
|---------|------|-----------|
| `http_request_duration_seconds` | Histogram | Duração das requisições HTTP |
| `http_requests_total` | Counter | Total de requisições por status |
| `process_cpu_seconds_total` | Counter | Tempo de CPU do processo |
| `process_resident_memory_bytes` | Gauge | Memória residente do processo |

#### Métricas do modelo (via `app/metrics.py`)

| Métrica | Tipo | Descrição |
|---------|------|-----------|
| `model_predictions_total` | Counter | Total de predições por classificação (`at_risk` / `not_at_risk`) |
| `model_prediction_probability` | Histogram | Distribuição das probabilidades de risco |
| `model_prediction_latency_seconds` | Histogram | Tempo de inferência do modelo |
| `model_batch_size` | Histogram | Tamanho dos lotes de predição |
| `model_at_risk_rate` | Gauge | Taxa atual de estudantes em risco |
| `model_drift_detected` | Gauge | Indica se drift foi detectado (0 ou 1) |
| `model_drift_features_count` | Gauge | Número de features com drift |
| `model_drift_total_features` | Gauge | Total de features monitoradas |

## 8. Considerações Éticas

- As predições devem ser usadas como **ferramenta de apoio**, não como decisão final
- Recomenda-se análise complementar por profissionais de educação
- O modelo visa **identificar** estudantes que precisam de mais atenção, não rotulá-los
- Atualizações periódicas são necessárias para manter a relevância do modelo
- **Sobre o uso de gênero como feature**: a variável `genero` foi incluída no modelo porque pode indicar diferenças nos padrões de defasagem entre grupos. O objetivo não é discriminar, mas permitir que o modelo capture essas diferenças para gerar predições mais precisas e direcionar intervenções de forma mais eficaz. Recomenda-se monitorar periodicamente se essa feature introduz viés indesejado nas predições.

## 9. Limitações

- Modelo treinado com dados específicos da Associação Passos Mágicos
- Performance pode variar para contextos educacionais diferentes
- Requer features específicas do programa para predições precisas (indicadores IAA, IEG, IPS, IDA, IPV e notas)
- Dados com poucas features preenchidas resultam em predições com menor confiança

## 10. Comandos Úteis

```bash
# Instalar dependências
uv sync --all-extras

# Gerar o arquivo requirements.txt com todas as bibliotecas externas e suas versões específicas necessárias para o projeto
uv pip compile pyproject.toml -o requirements.txt

# Treinar modelo
uv run python -m src.pipeline

# Iniciar API
uv run uvicorn app.main:app --port 8000

# Executar testes
uv run pytest

# Executar testes com cobertura
uv run pytest --cov=src --cov=app --cov-report=term-missing

# Subir toda a stack com Docker Compose (API + Prometheus + Grafana)
docker-compose up --build

# Subir em segundo plano
docker-compose up --build -d

# Parar todos os serviços
docker-compose down

# Build e execução apenas da API (sem monitoramento)
docker build -t passos-magicos-api .
docker run -p 8000:8000 passos-magicos-api
```

## 11. Autores

Projeto desenvolvido para o Datathon - Pós Tech FIAP

---

**Associação Passos Mágicos** - Transformando vidas através da educação desde 1992.
