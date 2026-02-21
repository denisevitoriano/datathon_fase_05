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

### Stack Tecnológica

| Componente | Tecnologia |
|------------|------------|
| Linguagem | Python 3.10+ |
| Gerenciador de Pacotes | uv |
| Frameworks ML | scikit-learn, pandas, numpy |
| API | FastAPI |
| Serialização | joblib |
| Testes | pytest (119 testes, 74% cobertura) |
| Empacotamento | Docker |
| Monitoramento | logging + drift detection |

## 2. Estrutura do Projeto

```
passos-magicos-ml/
├── app/                          # API FastAPI
│   ├── main.py                   # Aplicação principal
│   ├── routes.py                 # Endpoints de predição
│   ├── monitoring_routes.py      # Endpoints de monitoramento
│   ├── schemas.py                # Schemas Pydantic
│   └── model/                    # Artefatos do modelo
│       ├── predictor.py          # Lógica de predição
│       ├── model.joblib          # Modelo treinado
│       ├── preprocessor.joblib   # Preprocessador
│       └── features_config.json  # Configuração de features
│
├── src/                          # Pipeline de ML
│   ├── preprocessing.py          # Limpeza e preparação de dados
│   ├── feature_engineering.py    # Criação de features
│   ├── train.py                  # Treinamento do modelo
│   ├── evaluate.py               # Avaliação e métricas
│   ├── monitoring.py             # Detecção de drift
│   ├── pipeline.py               # Pipeline principal
│   └── utils.py                  # Funções auxiliares
│
├── tests/                        # Testes unitários
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_model.py
│   ├── test_evaluate.py
│   ├── test_monitoring.py
│   ├── test_utils.py
│   └── test_api.py
│
├── data/
│   ├── raw/                      # Dados brutos
│   └── processed/                # Dados processados
│
├── logs/                         # Logs e relatórios
├── notebooks/                    # Jupyter notebooks
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml                # Configuração do projeto e dependências
└── README.md
```

## 3. Instruções de Deploy

### Pré-requisitos
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes)
- Docker (opcional)

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
cd passos-magicos-ml

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
# http://localhost:8000/docs
```

### Deploy com Docker

```bash
# Build da imagem
docker build -t passos-magicos-api .

# Execute o container
docker run -p 8000:8000 passos-magicos-api

# Ou use docker-compose
docker-compose up -d
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

## 4. Exemplos de Chamadas à API

### Predição Individual

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "genero": "Masculino",
    "instituicao": "Pública",
    "pedra": "Topázio"
  }'
```

**Resposta:**
```json
{
  "at_risk": false,
  "risk_probability": 0.12,
  "risk_level": "Baixo",
  "confidence": 0.75
}
```

### Predição em Lote

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "students": [
      {"inde": 7.5, "iaa": 8.0, "ieg": 7.0, "idade": 12},
      {"inde": 6.0, "iaa": 5.5, "ieg": 6.0, "idade": 14}
    ]
  }'
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
  "model_type": "logistic",
  "timestamp": "2026-01-19T16:00:12.141970"
}
```

### Informações do Modelo

```bash
curl http://localhost:8000/model/info
```

### Python

```python
import requests

# Predição individual
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "inde": 7.5,
        "iaa": 8.0,
        "ieg": 7.0,
        "idade": 12,
        "genero": "Masculino"
    }
)
print(response.json())
```

## 5. Etapas do Pipeline de Machine Learning

### 5.1 Pré-processamento dos Dados
- Carregamento dos dados do Excel (múltiplas planilhas: PEDE2022, PEDE2023, PEDE2024)
- Padronização de nomes de colunas entre anos
- Remoção de colunas duplicadas
- Tratamento de valores faltantes
- Conversão de tipos de dados
- Remoção de colunas com vazamento de dados (IAN, fase_ideal)

### 5.2 Engenharia de Features
- **Features numéricas**: INDE, IAA, IEG, IPS, IPP, IDA, IPV, Mat, Por, Ing, idade, ano_ingresso
- **Features categóricas**: gênero, instituição, pedra
- **Features derivadas**:
  - Média das notas (mat, por, ing)
  - Anos no programa
  - Média dos indicadores
  - Razões entre indicadores (ida/ieg, ipv/ips)

### 5.3 Treinamento e Validação
- **Target**: Classificação binária (em risco vs. não em risco)
  - Em risco: defasagem < 0 (atrasado em relação à fase ideal)
  - Não em risco: defasagem >= 0
- **Modelos avaliados**:
  - Random Forest
  - Gradient Boosting
  - Logistic Regression (selecionado automaticamente)
- **Validação**: Cross-validation estratificada (5 folds)

### 5.4 Seleção de Modelo
- Comparação automática de modelos
- Seleção baseada em F1-Score
- Balanceamento de classes com `class_weight='balanced'`

### 5.5 Métricas de Avaliação
- **Accuracy**: Proporção de predições corretas
- **Precision**: Proporção de verdadeiros positivos entre preditos positivos
- **Recall**: Proporção de verdadeiros positivos identificados
- **F1-Score**: Média harmônica de precision e recall
- **ROC-AUC**: Área sob a curva ROC

## 6. Monitoramento

### Endpoints de Monitoramento

| Endpoint | Descrição |
|----------|-----------|
| `/monitoring/dashboard` | Dashboard com sumário de predições |
| `/monitoring/drift-status` | Status atual de drift |
| `/monitoring/prediction-history` | Histórico de predições |
| `/monitoring/metrics-summary` | Métricas do modelo |

### Detecção de Drift
- **Kolmogorov-Smirnov Test**: Detecta mudanças na distribuição
- **Population Stability Index (PSI)**: Quantifica a magnitude do drift
- Threshold configurável (default: p-value < 0.05 ou PSI > 0.2)

## 7. Variável Target e Interpretação

### Defasagem Escolar
A variável `defasagem` representa a diferença entre a fase atual do estudante e a fase ideal baseada na idade:

| Defasagem | Interpretação |
|-----------|---------------|
| -2 ou menos | Alto risco - 2+ anos atrasado |
| -1 | Risco moderado - 1 ano atrasado |
| 0 | Adequado - na fase correta |
| +1 ou mais | Adiantado |

### Classificação de Risco (Saída do Modelo)

| Probabilidade | Nível de Risco | Recomendação |
|---------------|----------------|--------------|
| < 0.3 | Baixo | Acompanhamento regular |
| 0.3 - 0.7 | Médio | Atenção especial |
| > 0.7 | Alto | Intervenção prioritária |

## 8. Considerações Éticas

- As predições devem ser usadas como **ferramenta de apoio**, não como decisão final
- Recomenda-se análise complementar por profissionais de educação
- O modelo visa **identificar** estudantes que precisam de mais atenção, não rotulá-los
- Atualizações periódicas são necessárias para manter a relevância do modelo

## 9. Limitações

- Modelo treinado com dados específicos da Associação Passos Mágicos
- Performance pode variar para contextos educacionais diferentes
- Requer features específicas do programa para predições precisas
- Dados históricos (Pedra anterior, INDE anterior) melhoram a precisão

## 10. Guia de Testes Manuais

### Passo 1: Preparar o Ambiente

```bash
# Instalar dependências
uv sync --all-extras

# Verificar se os dados estão no lugar correto
ls data/raw/
# Deve mostrar: BASE DE DADOS PEDE 2024 - DATATHON.xlsx
```

### Passo 2: Treinar o Modelo

```bash
# Executar pipeline de treinamento
uv run python -m src.pipeline
```

**Saída esperada:**
- Logs mostrando carregamento dos dados (1156 linhas)
- Comparação de 3 modelos (random_forest, gradient_boosting, logistic)
- Seleção automática do melhor modelo
- Salvamento dos artefatos em `app/model/`

**Verificar artefatos criados:**
```bash
ls app/model/
# Deve mostrar: model.joblib, preprocessor.joblib, features_config.json
```

### Passo 3: Iniciar a API

```bash
# Em um terminal, inicie a API
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Saída esperada:**
```
INFO:     Carregando modelo e preprocessador...
INFO:     Artefatos do modelo carregados com sucesso
INFO:     Modelo carregado com sucesso!
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Passo 4: Testar Endpoints (em outro terminal)

#### 4.1 Health Check
```bash
curl http://localhost:8000/health
```

**Resposta esperada:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "logistic",
  "timestamp": "2026-01-19T..."
}
```

#### 4.2 Endpoint Raiz
```bash
curl http://localhost:8000/
```

**Resposta esperada:** Lista de endpoints disponíveis

#### 4.3 Predição Individual (Estudante SEM risco)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inde": 8.5, "iaa": 9.0, "ieg": 8.0, "idade": 10, "genero": "Feminino"}'
```

**Resposta esperada:** `at_risk: false`, `risk_level: "Baixo"`

#### 4.4 Predição Individual (Estudante COM risco)
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"inde": 4.0, "iaa": 3.5, "ieg": 4.0, "idade": 15, "genero": "Masculino"}'
```

**Resposta esperada:** `at_risk: true`, `risk_level: "Alto"`

#### 4.5 Predição em Lote
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "students": [
      {"inde": 8.0, "iaa": 8.5, "ieg": 7.5, "idade": 11},
      {"inde": 5.0, "iaa": 4.0, "ieg": 5.0, "idade": 14},
      {"inde": 7.0, "iaa": 7.0, "ieg": 7.0, "idade": 12}
    ]
  }'
```

**Resposta esperada:** 3 predições com `total_processed: 3`

#### 4.6 Informações do Modelo
```bash
curl http://localhost:8000/model/info
```

**Resposta esperada:** Tipo do modelo, métricas e informações de features

#### 4.7 Lista de Features
```bash
curl http://localhost:8000/features
```

**Resposta esperada:** Lista de features numéricas e categóricas

#### 4.8 Dashboard de Monitoramento
```bash
curl http://localhost:8000/monitoring/dashboard
```

**Resposta esperada:** Sumário de predições e status de drift

### Passo 5: Acessar Documentação Interativa

Abra no navegador:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

Você pode testar todos os endpoints diretamente pela interface.

### Passo 6: Executar Testes Automatizados

```bash
# Parar a API (Ctrl+C) e executar testes
uv run pytest -v
```

**Saída esperada:** 119 testes passando

```bash
# Com cobertura detalhada
uv run pytest --cov=src --cov=app --cov-report=term-missing
```

**Saída esperada:** ~74% de cobertura

### Passo 7: Testar com Docker (Opcional)

```bash
# Build da imagem
docker build -t passos-magicos-api .

# Executar container
docker run -p 8000:8000 passos-magicos-api

# Testar (mesmos comandos curl do Passo 4)
curl http://localhost:8000/health
```

### Checklist de Validação

- [ ] Ambiente instalado com `uv sync`
- [ ] Modelo treinado com sucesso
- [ ] Artefatos salvos em `app/model/`
- [ ] API inicia sem erros
- [ ] Health check retorna `status: healthy`
- [ ] Predição individual funciona
- [ ] Predição em lote funciona
- [ ] Swagger UI acessível em `/docs`
- [ ] 119 testes passando
- [ ] Docker build funciona (opcional)

## 11. Comandos Úteis

```bash
# Instalar dependências
uv sync --all-extras

# Treinar modelo
uv run python -m src.pipeline

# Iniciar API
uv run uvicorn app.main:app --port 8000

# Executar testes
uv run pytest

# Executar testes com cobertura
uv run pytest --cov=src --cov=app --cov-report=term-missing

# Build Docker
docker build -t passos-magicos-api .

# Executar container
docker run -p 8000:8000 passos-magicos-api
```

## 12. Autores

Projeto desenvolvido para o Datathon - Pós Tech FIAP

---

**Associação Passos Mágicos** - Transformando vidas através da educação desde 1992.
