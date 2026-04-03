# CLAUDE.md — Bíblia Arquitetural: MLP Value Investing para AGRO3

> Este documento é a fonte canônica de verdade para todas as decisões de design, arquitetura e implementação deste projeto. Toda interação com o assistente de IA deve respeitar estas diretrizes sem exceção.

---

## 1. Visão Geral do Projeto

### 1.1 O Problema Fundamental (Por que não Regressão?)

Prever o preço exato de uma ação amanhã é, na prática, um problema de regressão sobre ruído. Mercados semi-eficientes precificam informação rapidamente, e o sinal-ruído de séries temporais de preços é notoriamente baixo. **Portanto, não construiremos um regressor de preços.**

O problema real que um investidor de longo prazo baseado em Value Investing precisa resolver é de natureza classificatória:

> *"Dado o estado atual dos fundamentos, do mercado e da narrativa da empresa, o ativo representa uma oportunidade de aporte, indiferença, ou um sinal de saída?"*

### 1.2 A Solução: Classificação Multiclasse com Contexto Temporal

O modelo produzirá um único output com 3 classes:

| Label | Classe   | Definição Operacional                                                   |
|-------|----------|-------------------------------------------------------------------------|
| `1`   | COMPRAR  | Retorno Total (12m) superou o CDI do período por margem > threshold     |
| `0`   | AGUARDAR | Retorno Total (12m) ficou dentro da banda de indiferença vs. CDI        |
| `-1`  | VENDER   | Retorno Total (12m) foi inferior ao CDI do período (custo de oportunidade) |

### 1.3 O Desafio Técnico: MLP sem Memória Nativa

Uma MLP pura recebe um vetor estático de features. Ela não possui estado recorrente (como uma LSTM) nem mecanismo de atenção temporal (como um Transformer). O desafio central é:

**Como fazer uma MLP aprender a dinâmica temporal de uma série financeira?**

### 1.4 A Solução: Janela Deslizante (Sliding Window) na Camada de Entrada

A técnica de **Sliding Window** transforma o problema temporal em um problema estático de alta dimensionalidade que a MLP pode aprender:

```
Série temporal: [f_t-W, f_t-W+1, ..., f_t-1, f_t]
                |___________________________________|
                         Janela de W períodos
                                  ↓
           Vetor de entrada achatado: R^(W × num_features)
```

- **W (window size)**: Hiperparâmetro a ser otimizado. Valores iniciais a explorar: `[12, 24, 52]` semanas.
- **Sem data leakage**: A janela em `t` usa apenas dados até `t`. O label em `t` usa retorno de `t+1` a `t+52`.
- A MLP aprende implicitamente padrões de momentum, reversão à média e ciclos de earnings dentro da janela.

### 1.5 Construção do Gabarito (Target Y)

O label histórico é construído da seguinte forma:

```python
# Pseudo-código conceitual — implementação real em src/feature_engineering/target_builder.py
retorno_total_12m = (preco_t_52 - preco_t + dividendos_acumulados) / preco_t
cdi_12m = cdi_acumulado_no_periodo(t, t+52)

margem = retorno_total_12m - cdi_12m

if margem > THRESHOLD_COMPRA:    # ex: +5pp acima do CDI
    label = 1   # COMPRAR
elif margem < THRESHOLD_VENDA:   # ex: -5pp abaixo do CDI
    label = -1  # VENDER
else:
    label = 0   # AGUARDAR
```

**Justificativa dos thresholds**: O CDI representa o custo de oportunidade brasileiro (ativo livre de risco). Superar o CDI em +5pp liquida o prêmio de risco de renda variável justificando alocação em equity.

---

## 2. Regra de Negócio e Feature Engineering

### 2.1 O Diferencial Competitivo da AGRO3: Real Estate Rural

A maioria dos modelos quantitativos trata a AGRO3 como uma commodity play (soja, milho, câmbio). **Esta é uma análise incompleta e superficial.** O verdadeiro motor de criação de valor da BrasilAgro é seu flywheel imobiliário rural:

```
Aquisição de terras brutas (low-cost)
           ↓
  Desenvolvimento agrícola e infraestrutura
           ↓
    Valorização dos ativos fundiários
           ↓
  Venda com alta TIR (15-30% aa histórico)
           ↓
   Distribuição de dividendos expressivos
           ↓
  Reciclagem de capital para novas aquisições
```

**Implicação para o modelo**: As features mais preditivas provavelmente não são o preço da soja hoje, mas sim:
- A qualidade do pipeline de aquisições (mencionada em fatos relevantes)
- O valor de mercado das terras vs. valor contábil (P/VPA de terras)
- O ritmo de desinvestimentos e o timing dos ciclos de vendas
- A narrativa de gestão sobre oportunidades de aquisição

### 2.2 Categorias de Features

#### 2.2.1 Features de Preço e Volume (Numéricas — `yfinance`)

```
grupo: price_action
- close_adj          : Preço de fechamento ajustado
- volume             : Volume financeiro diário
- return_1w          : Retorno semanal
- return_4w          : Retorno mensal
- return_13w         : Retorno trimestral
- volatility_4w      : Volatilidade realizada (4 semanas)
- rsi_14             : RSI 14 períodos
- price_to_52w_high  : Distância ao máximo de 52 semanas (proxy de momentum)
```

#### 2.2.2 Features Fundamentalistas (Trimestrais — `fundamentus` + RI)

```
grupo: fundamentals
- p_vpa              : Preço / Valor Patrimonial por Ação
- ev_ebitda           : Enterprise Value / EBITDA
- dividend_yield     : DY dos últimos 12 meses
- roe                : Return on Equity
- net_debt_ebitda    : Alavancagem
- gross_margin       : Margem Bruta
- land_bank_value    : Valor estimado do banco de terras (extraído de RI — *ver seção NLP*)
- land_sales_tir     : TIR média declarada nas vendas de terras (extraído de RI)
```

#### 2.2.3 Features Macroeconômicas (Semanais — BCB/FRED)

```
grupo: macro
- cdi_rate           : Taxa CDI anualizada (custo de oportunidade)
- usd_brl            : Taxa de câmbio USD/BRL
- soy_price_usd      : Preço da soja em Chicago (CBOT)
- corn_price_usd     : Preço do milho em Chicago (CBOT)
- igpm_12m           : IGPM acumulado 12m (inflação de terras e aluguéis rurais)
- selic_real         : Selic real (Selic - IPCA projetado)
```

#### 2.2.4 Features de Sentimento NLP (Trimestrais — PDFs da B3/RI)

```
grupo: sentiment_nlp
- sentiment_land_acq    : Score de sentimento focado em aquisições de terras
- sentiment_land_sale   : Score de sentimento focado em vendas/desinvestimentos
- sentiment_guidance    : Score de sentimento sobre guidance de gestão
- sentiment_macro_agro  : Score de sentimento sobre cenário macro agrícola
- release_urgency       : Proxy de urgência/materialidade do fato relevante
```

### 2.3 Pipeline NLP: Texto → Feature Numérica

```
PDF (B3/RI) → Extração de Texto (pdfplumber)
                    ↓
          Limpeza e Chunking por seção
                    ↓
     Encoder Transformer (BERTimbau ou FinBERT-PT-BR)
                    ↓
        Classificação de sentimento por categoria
                    ↓
    Score normalizado [-1, 1] por categoria por release
                    ↓
   Interpolação temporal para séries semanais contínuas
                    ↓
         Incorporação como features no dataset principal
```

**Modelos NLP candidatos** (em ordem de preferência):
1. `neuralmind/bert-base-portuguese-cased` (BERTimbau) — Pré-treinado em português
2. `rufimelo/bert-base-portuguese-cased-sts` — Para similaridade semântica
3. Fine-tuning supervisionado caso tenhamos labels manuais suficientes

---

## 3. Stack Tecnológico

### 3.1 Regras Invioláveis de Ambiente

| Ferramenta       | Versão mínima | Uso                                           |
|------------------|---------------|-----------------------------------------------|
| Python           | 3.12+         | Linguagem principal                           |
| `uv`             | latest        | **ÚNICO** gerenciador de pacotes e ambientes  |
| PyTorch          | 2.3+          | Framework de Deep Learning (MLP)              |
| `transformers`   | 4.40+         | Modelos NLP (Hugging Face)                    |
| `yfinance`       | 0.2+          | Dados de mercado e corporativos               |
| `pandas`         | 2.2+          | Manipulação de dados tabulares                |
| `numpy`          | 1.26+         | Operações vetoriais                           |
| `scikit-learn`   | 1.4+          | Pré-processamento, métricas, baseline models  |
| `pdfplumber`     | 0.10+         | Extração de texto de PDFs                     |
| `pytest`         | 8.0+          | Framework de testes                           |
| `ruff`           | latest        | Linter e formatter                            |

### 3.2 Proibições Explícitas

- **NUNCA** usar `pip install` diretamente. Sempre `uv pip install` ou adicionar ao `pyproject.toml`.
- **NUNCA** usar `conda` ou `virtualenv` — apenas `uv venv`.
- **NUNCA** commitar o diretório `.venv/`.
- **NUNCA** usar Jupyter Notebooks como artefatos de produção. Notebooks são permitidos **apenas** em `notebooks/exploratory/` para EDA inicial e são descartáveis.

---

## 4. Arquitetura do Projeto

### 4.1 Estrutura de Diretórios

```
investing-AGRO3/
│
├── CLAUDE.md                    # Este arquivo — fonte de verdade arquitetural
├── pyproject.toml               # Dependências e configuração do projeto (uv)
├── README.md                    # Documentação pública do projeto
├── .gitignore
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_ingestion/          # Módulo 1: Coleta de dados brutos
│   │   ├── __init__.py
│   │   ├── market_data.py       # yfinance: cotações, dividendos AGRO3
│   │   ├── fundamentals.py      # fundamentus: múltiplos fundamentalistas
│   │   ├── macro_data.py        # BCB API, FRED: CDI, câmbio, commodities
│   │   └── pdf_downloader.py    # Download de PDFs da B3 e RI BrasilAgro
│   │
│   ├── feature_engineering/     # Módulo 2: Transformação e construção de features
│   │   ├── __init__.py
│   │   ├── target_builder.py    # Construção do label Y (Comprar/Aguardar/Vender)
│   │   ├── technical_features.py # RSI, volatilidade, momentum
│   │   ├── fundamental_features.py # Normalização e alinhamento temporal de fundamentos
│   │   ├── nlp_pipeline.py      # Texto PDF → Score de sentimento numérico
│   │   └── sliding_window.py    # Transformação de séries em janelas para MLP
│   │
│   ├── models/                  # Módulo 3: Arquiteturas e treinamento PyTorch
│   │   ├── __init__.py
│   │   ├── mlp.py               # Definição da arquitetura MLP (nn.Module)
│   │   ├── dataset.py           # AgRo3Dataset (torch.utils.data.Dataset)
│   │   ├── trainer.py           # Loop de treinamento, validação, early stopping
│   │   └── hyperparameter_search.py # Otimização de hiperparâmetros
│   │
│   ├── evaluation/              # Módulo 4: Métricas e backtesting
│   │   ├── __init__.py
│   │   ├── metrics.py           # Precision, recall, F1 por classe; confusion matrix
│   │   ├── backtester.py        # Simulação histórica de portfólio com sinais do modelo
│   │   └── reporting.py         # Geração de relatórios e visualizações
│   │
│   └── utils/                   # Utilitários transversais
│       ├── __init__.py
│       ├── logger.py            # Configuração de logging estruturado
│       ├── config.py            # Carregamento de configurações (YAML/env)
│       └── validators.py        # Validação de DataFrames e schemas de dados
│
├── config/
│   ├── model_config.yaml        # Hiperparâmetros do modelo MLP
│   └── pipeline_config.yaml     # Configurações do pipeline de dados
│
├── data/
│   ├── raw/                     # Dados brutos (nunca modificar)
│   │   ├── market/
│   │   ├── fundamentals/
│   │   ├── macro/
│   │   └── pdfs/
│   ├── processed/               # Dados após feature engineering
│   └── models/                  # Artefatos de modelos treinados (.pt)
│
├── notebooks/
│   └── exploratory/             # EDA descartável — NÃO são artefatos de produção
│
└── tests/
    ├── unit/
    │   ├── test_target_builder.py
    │   ├── test_sliding_window.py
    │   └── test_mlp_forward_pass.py
    └── integration/
        └── test_full_pipeline.py
```

### 4.2 Arquitetura da MLP (`src/models/mlp.py`)

```python
# Arquitetura de referência — detalhes serão refinados durante implementação

Input Layer:  R^(W × num_features)   # ex: 52 semanas × 25 features = 1300 neurônios
                    ↓
Hidden Layer 1: 512 neurônios + BatchNorm + ReLU + Dropout(0.3)
                    ↓
Hidden Layer 2: 256 neurônios + BatchNorm + ReLU + Dropout(0.3)
                    ↓
Hidden Layer 3: 128 neurônios + BatchNorm + ReLU + Dropout(0.2)
                    ↓
Hidden Layer 4:  64 neurônios + BatchNorm + ReLU + Dropout(0.2)
                    ↓
Output Layer:     3 neurônios (logits para CrossEntropyLoss)
                    ↓
            Softmax → P(COMPRAR), P(AGUARDAR), P(VENDER)
```

**Decisões de design justificadas**:
- `BatchNorm` antes da ativação: estabiliza gradientes em séries financeiras com alta variância
- `Dropout` crescente para dentro: regularização progressiva
- `CrossEntropyLoss` com `class_weight`: dados financeiros são inerentemente desbalanceados
- Otimizador: `AdamW` com `weight_decay=1e-4` e `CosineAnnealingLR`

### 4.3 Princípios SOLID Aplicados ao Projeto

| Princípio | Aplicação Concreta                                                                      |
|-----------|-----------------------------------------------------------------------------------------|
| **S**RP   | Cada módulo tem uma única responsabilidade (ingestão ≠ feature ≠ modelo ≠ avaliação)    |
| **O**CP   | Novas fontes de dados = novo arquivo em `data_ingestion/`, sem modificar existentes     |
| **L**SP   | Todos os `Dataset` são subclasses substituíveis de `torch.utils.data.Dataset`           |
| **I**SP   | Interfaces de ingestão separadas: `MarketDataFetcher`, `FundamentalsFetcher`, etc.      |
| **D**IP   | `Trainer` depende de abstrações (`nn.Module`), não de `MLP` diretamente                |

---

## 5. Padrões de Código Obrigatórios

### 5.1 Type Hinting

**Todo** código de produção em `src/` deve ter type hints completos. Sem exceções.

```python
# CORRETO
def build_sliding_windows(
    df: pd.DataFrame,
    window_size: int,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    ...

# PROIBIDO
def build_sliding_windows(df, window_size, feature_cols, target_col):
    ...
```

### 5.2 Tratamento de Erros em I/O

Toda chamada de rede, leitura de arquivo ou acesso a API deve ter tratamento explícito:

```python
# Padrão obrigatório para chamadas de rede
import logging
from typing import Optional
import yfinance as yf

logger = logging.getLogger(__name__)

def fetch_agro3_data(
    start_date: str,
    end_date: str,
    retries: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Busca dados históricos da AGRO3 via yfinance com retry automático.

    Args:
        start_date: Data inicial no formato 'YYYY-MM-DD'.
        end_date: Data final no formato 'YYYY-MM-DD'.
        retries: Número máximo de tentativas em caso de falha de rede.

    Returns:
        DataFrame com OHLCV ou None em caso de falha persistente.
    """
    for attempt in range(retries):
        try:
            ticker = yf.Ticker("AGRO3.SA")
            df = ticker.history(start=start_date, end=end_date)
            if df.empty:
                raise ValueError("Resposta vazia do yfinance para AGRO3.SA")
            return df
        except Exception as e:
            logger.warning(f"Tentativa {attempt + 1}/{retries} falhou: {e}")
            if attempt == retries - 1:
                logger.error("Todas as tentativas esgotadas. Retornando None.")
                return None
```

### 5.3 Prevenção de Data Leakage

Esta é a regra mais crítica de qualidade do modelo. **Qualquer violação invalida o backtest.**

```
REGRA: Para uma amostra com timestamp t, NENHUMA feature pode
       conter informação de instantes t' > t.

Verificações obrigatórias:
  1. Sliding window: janela termina em t, nunca além.
  2. Normalização: scaler fitado APENAS no train set, aplicado no val/test.
  3. Features fundamentalistas: usar valor do trimestre ANTERIOR ao de t.
  4. Features NLP: usar releases publicados ANTES de t.
  5. Target Y: usa dados de t+1 a t+52 — correto por design (é o que queremos prever).
```

### 5.4 Divisão Temporal do Dataset

```
|-------- TRAIN (60%) ---------|-- VAL (20%) --|-- TEST (20%) --|
2006                         2018            2022             2025
                   ↑                ↑
            Walk-Forward         Sem shuffle
            Cross-Validation     (temporal integrity)
```

**Proibido**: `train_test_split` com `shuffle=True` em dados temporais.

---

## 6. Fluxo de Execução do Pipeline Completo

```
1. uv run python -m src.data_ingestion.market_data      → data/raw/market/
2. uv run python -m src.data_ingestion.fundamentals     → data/raw/fundamentals/
3. uv run python -m src.data_ingestion.macro_data       → data/raw/macro/
4. uv run python -m src.data_ingestion.pdf_downloader   → data/raw/pdfs/
                              ↓
5. uv run python -m src.feature_engineering.nlp_pipeline        → NLP scores
6. uv run python -m src.feature_engineering.technical_features  → indicadores técnicos
7. uv run python -m src.feature_engineering.fundamental_features → fundamentos alinhados
8. uv run python -m src.feature_engineering.target_builder      → labels Y
9. uv run python -m src.feature_engineering.sliding_window      → data/processed/
                              ↓
10. uv run python -m src.models.trainer                         → data/models/mlp_v1.pt
                              ↓
11. uv run python -m src.evaluation.backtester                  → relatório de performance
```

---

## 7. Métricas de Avaliação

O modelo será avaliado por múltiplas métricas complementares:

| Métrica                  | Justificativa                                                             |
|--------------------------|---------------------------------------------------------------------------|
| **F1-Score Macro**       | Métrica primária — penaliza desequilíbrio entre classes                   |
| **Precision (COMPRAR)**  | Custo de falso positivo: alocar capital em ativo que vai performar mal    |
| **Recall (VENDER)**      | Custo de falso negativo: não vender antes de uma queda acentuada          |
| **Sharpe Ratio (BT)**    | Performance ajustada ao risco no backtest histórico                       |
| **Alpha vs. CDI (BT)**   | Alpha gerado pelo modelo vs. benchmark (buy-and-hold CDI)                 |
| **Max Drawdown (BT)**    | Pior sequência de perdas — proxy de risco real para o investidor          |

---

## 8. Checklist de Implementação (Ordem de Desenvolvimento)

- [ ] **Fase 0**: Setup (ambiente `uv`, `pyproject.toml`, estrutura de diretórios, `.gitignore`)
- [ ] **Fase 1**: `data_ingestion` — Coleta de dados numéricos (market + fundamentals + macro)
- [ ] **Fase 2**: `feature_engineering` — Target builder + features técnicas e fundamentalistas
- [ ] **Fase 3**: `feature_engineering/sliding_window.py` — Dataset PyTorch com janelas
- [ ] **Fase 4**: `models/mlp.py` + `models/trainer.py` — Arquitetura e loop de treinamento
- [ ] **Fase 5**: `evaluation/metrics.py` — Validação estatística do modelo
- [ ] **Fase 6**: `data_ingestion/pdf_downloader.py` + `feature_engineering/nlp_pipeline.py` — NLP
- [ ] **Fase 7**: `evaluation/backtester.py` — Simulação histórica completa
- [ ] **Fase 8**: Hiperparameter search + análise de feature importance
