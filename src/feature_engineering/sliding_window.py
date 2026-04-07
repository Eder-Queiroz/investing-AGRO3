"""
Sliding Window — Fase 3: Transformação de séries temporais em janelas para MLP.

Este módulo é a ponte entre o DataFrame tabular da Fase 2 e o mundo do Deep Learning.
É um módulo **pure Python/numpy/pandas** — sem dependência de PyTorch — para garantir
testabilidade independente.

## Fluxo de dados

    features_weekly.parquet (989 × 38)
        ↓ load_features_parquet()
    pd.DataFrame com DatetimeIndex W-FRI
        ↓ compute_valid_indices()
    np.ndarray de posições inteiras válidas (≈834 para W=52)
        ↓ split_indices_chronological()
    train_idx, val_idx, test_idx
        ↓ build_windows() × 3
    X: (N, W*F), y: (N,) com labels {-1, 0, 1}
        ↓ fit_scaler(X_train) + scaler.transform()
    X normalizado (float64)
        ↓ remap_labels()
    y: {0, 1, 2} pronto para CrossEntropyLoss

## MODEL_FEATURE_COLS — 23 features estacionárias (ordem é load-bearing)

A ordem desta lista define o layout do vetor de entrada da MLP. Alterar
os elementos ou sua ordem invalida qualquer checkpoint de modelo salvo.
Documente toda mudança com a versão do dataset correspondente.

## Geometria do Flattening

Para W semanas e F=23 features, uma amostra é a matriz (W, F) achatada em
C-order (row-major):

    [semana_{t-W+1}_feat_0, ..., semana_{t-W+1}_feat_22,
     semana_{t-W+2}_feat_0, ..., semana_{t-W+2}_feat_22,
     ...
     semana_t_feat_0,       ..., semana_t_feat_22]

Posições 0..F-1 = semana mais antiga; posições -F..-1 = semana mais recente (t).
A MLP aprende pesos dependentes de posição temporal: o neurônio na posição k
da entrada sempre corresponde à mesma feature da mesma posição relativa na janela.

## Anti-Leakage

O `StandardScaler` deve ser fitado APENAS nos dados de treino (`fit_scaler`).
Persista o scaler junto ao checkpoint do modelo para uso em inferência.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.config import load_model_config
from src.utils.logger import get_logger
from src.utils.validators import validate_columns, validate_no_future_leakage

logger: logging.Logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Colunas de entrada do modelo — 23 features estacionárias.
# ---------------------------------------------------------------------------
# Excluídos explicitamente (non-stationary):
#   - Preços/volume brutos: open_adj, high_adj, low_adj, close_adj, volume
#   - Múltiplos fundamentalistas brutos: p_vpa, ev_ebitda, roe, net_debt_ebitda,
#     gross_margin, dividend_yield  (temos as versões z-score abaixo)
#   - Macro raw não-estacionários: usd_brl → log_return_usd_brl_4w;
#     soy_price_usd → log_return_soy_4w; corn_price_usd → log_return_corn_4w
MODEL_FEATURE_COLS: list[str] = [
    # --- Technical features (12) — já estacionárias por construção ---
    "log_return_1w",
    "log_return_4w",
    "log_return_13w",
    "volatility_4w",
    "rsi_14",
    "price_to_52w_high",
    "volume_zscore_4w",
    "log_return_soy_4w",
    "log_return_corn_4w",
    "log_return_usd_brl_4w",
    "delta_cdi_4w",
    "delta_selic_real_4w",
    # --- Fundamental z-scores (6) — expanding z-score → ~N(0,1) ---
    "p_vpa_zscore",
    "ev_ebitda_zscore",
    "roe_zscore",
    "net_debt_ebitda_zscore",
    "gross_margin_zscore",
    "dividend_yield_zscore",
    # --- Macro rates I(0) (5) — taxas de política/inflação são mean-reverting no BR ---
    "cdi_rate",
    "selic_rate",
    "igpm",
    "ipca",
    "selic_real",
]

# Label remapping: {-1 → 0, 0 → 1, 1 → 2}
# SELL=0, HOLD=1, BUY=2
# Reversível: original = remapped - 1
_LABEL_OFFSET: int = 1

_DEFAULT_PARQUET_PATH: Path = _PROJECT_ROOT / "data" / "processed" / "features_weekly.parquet"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_features_parquet(
    parquet_path: Path | None = None,
) -> pd.DataFrame:
    """Carrega o Parquet gerado pela Fase 2 e valida seu schema.

    Args:
        parquet_path: Caminho para o arquivo Parquet. Se None, usa o caminho
            padrão em data/processed/features_weekly.parquet.

    Returns:
        DataFrame com DatetimeIndex W-FRI, 38 colunas (37 features + target).

    Raises:
        FileNotFoundError: Se o arquivo Parquet não existir.
        ValueError: Se o schema ou a ordenação temporal estiverem incorretos.
    """
    path = parquet_path or _DEFAULT_PARQUET_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Parquet da Fase 2 não encontrado: {path}. "
            "Execute src/feature_engineering/pipeline.py primeiro."
        )

    logger.info(f"Carregando Parquet: {path}")
    df = pd.read_parquet(path)

    validate_no_future_leakage(df, "index", context="sliding_window.load_features_parquet")
    validate_columns(
        df, MODEL_FEATURE_COLS + ["target"], context="sliding_window.load_features_parquet"
    )

    nan_summary = df[MODEL_FEATURE_COLS].isna().sum()
    nan_cols = nan_summary[nan_summary > 0]
    logger.info(
        f"Parquet carregado: {df.shape[0]} semanas × {df.shape[1]} colunas. "
        f"Colunas com NaN: {len(nan_cols)}"
    )
    if not nan_cols.empty:
        logger.debug(f"NaN por coluna de feature:\n{nan_cols.to_string()}")

    return df


def compute_valid_indices(
    df: pd.DataFrame,
    window_size: int,
    feature_cols: list[str],
    target_col: str = "target",
) -> np.ndarray:
    """Pré-computa as posições inteiras válidas para construção de janelas.

    Uma posição t é válida se e somente se:
    1. A janela [t-window_size+1, t] contém apenas valores finitos em
       todos as feature_cols (sem NaN nem Inf).
    2. O target em t não é pd.NA.

    AVISO: Esta função itera sobre O(n) posições. Com ~989 linhas e W=52,
    isso é trivialmente rápido (< 50ms). Nenhuma vetorização prematura.

    Args:
        df: DataFrame da Fase 2.
        window_size: Tamanho da janela em semanas (W).
        feature_cols: Colunas a incluir em cada janela.
        target_col: Nome da coluna de target (default: "target").

    Returns:
        Array de inteiros dtype np.intp, em ordem crescente, com as
        posições (iloc) finais de cada janela válida.
    """
    n = len(df)
    feature_values: np.ndarray = df[feature_cols].values  # (n, F) float64
    target_series: pd.Series = df[target_col]

    valid: list[int] = []

    for t in range(window_size - 1, n):
        # Extrai janela de features: linhas [t-W+1, t] inclusive
        window = feature_values[t - window_size + 1 : t + 1]  # (W, F)

        # Condição 1: todos os valores são finitos (rejeita NaN e Inf)
        if not np.isfinite(window).all():
            continue

        # Condição 2: target não é pd.NA — usa pd.isna() pois dtype é pd.Int8Dtype()
        if pd.isna(target_series.iloc[t]):
            continue

        valid.append(t)

    result = np.array(valid, dtype=np.intp)
    n_candidates = n - (window_size - 1)
    logger.info(
        f"Índices válidos: {len(result)}/{n_candidates} candidatos "
        f"(W={window_size}, F={len(feature_cols)})"
    )
    return result


def split_indices_chronological(
    valid_indices: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Divide os índices válidos em train/val/test cronologicamente.

    Os três conjuntos são disjuntos e não há embaralhamento. A integridade
    temporal é garantida: max(train) < min(val) < min(test).

    Nota: Janelas de val/test podem usar linhas do período de train como
    contexto histórico (W-1 lookback). Isso é correto por design — não
    há vazamento de labels futuros no sentido causal.

    Args:
        valid_indices: Array de índices válidos em ordem crescente.
        train_ratio: Proporção do conjunto de treino (ex: 0.60).
        val_ratio: Proporção do conjunto de validação (ex: 0.20).

    Returns:
        Tripla (train_idx, val_idx, test_idx) de arrays np.intp.

    Raises:
        ValueError: Se os ratios forem inválidos.
    """
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError(
            f"train_ratio e val_ratio devem ser positivos. "
            f"Recebido: train_ratio={train_ratio}, val_ratio={val_ratio}"
        )
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError(
            f"train_ratio + val_ratio deve ser < 1.0. Soma: {train_ratio + val_ratio:.4f}"
        )

    n = len(valid_indices)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = valid_indices[:train_end]
    val_idx = valid_indices[train_end:val_end]
    test_idx = valid_indices[val_end:]

    logger.info(
        f"Split cronológico — Train: {len(train_idx)} | "
        f"Val: {len(val_idx)} | Test: {len(test_idx)} | "
        f"Total: {len(train_idx) + len(val_idx) + len(test_idx)}"
    )

    if len(train_idx) > 0 and len(val_idx) > 0:
        assert int(train_idx[-1]) < int(val_idx[0]), (
            "Sobreposição temporal detectada entre train e val!"
        )
    if len(val_idx) > 0 and len(test_idx) > 0:
        assert int(val_idx[-1]) < int(test_idx[0]), (
            "Sobreposição temporal detectada entre val e test!"
        )

    return train_idx, val_idx, test_idx


def build_windows(
    df: pd.DataFrame,
    indices: np.ndarray,
    window_size: int,
    feature_cols: list[str],
    target_col: str = "target",
) -> tuple[np.ndarray, np.ndarray]:
    """Constrói arrays de janelas achatadas para entrada na MLP.

    Cada amostra é a matriz (W, F) da janela achatada em C-order (row-major),
    preservando a ordenação temporal:
        - X[i, :F] = features da semana mais antiga na janela
        - X[i, -F:] = features da semana mais recente (semana t)

    Args:
        df: DataFrame da Fase 2.
        indices: Posições finais das janelas (output de compute_valid_indices).
        window_size: Tamanho da janela W.
        feature_cols: Colunas a incluir (define F).
        target_col: Coluna de target.

    Returns:
        Tupla (X, y):
            - X: shape (N, W*F), dtype float64
            - y: shape (N,), dtype int8, valores em {-1, 0, 1}
    """
    n_samples = len(indices)
    n_features = len(feature_cols)
    flat_dim = window_size * n_features

    feature_values: np.ndarray = df[feature_cols].values  # (n_rows, F)
    target_values: pd.Series = df[target_col]

    # Pré-aloca para evitar múltiplas realocações
    X = np.empty((n_samples, flat_dim), dtype=np.float64)
    y = np.empty(n_samples, dtype=np.int8)

    for i, t in enumerate(indices):
        window = feature_values[t - window_size + 1 : t + 1]  # (W, F)
        X[i] = window.flatten(order="C")
        y[i] = int(target_values.iloc[t])

    # Invariante pós-construção: defense in depth
    if not np.isfinite(X).all():
        raise ValueError(
            "[build_windows] X contém valores não-finitos após construção. "
            "Verifique compute_valid_indices — os índices fornecidos não são todos válidos."
        )

    logger.debug(
        f"build_windows: X={X.shape}, y={y.shape}, labels únicos: {sorted(set(y.tolist()))}"
    )
    return X, y


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Fita um StandardScaler APENAS nos dados de treino.

    Cada um dos W*F neurônios de entrada recebe sua própria média e desvio
    padrão — o escaler captura a magnitude típica de cada feature em cada
    posição temporal dentro da janela.

    CRÍTICO: Este método deve ser chamado exatamente UMA vez, usando apenas
    X_train. O scaler resultante deve ser salvo junto ao checkpoint do modelo
    para uso em inferência (sem refitar nos dados de val/test/produção).

    Args:
        X_train: Array (N_train, W*F) com os dados de treino não-escalados.

    Returns:
        StandardScaler fitado. Use scaler.transform() em val e test.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)

    assert scaler.mean_ is not None and scaler.mean_.shape == (X_train.shape[1],), (
        f"Scaler com shape inesperado: {scaler.mean_.shape}"
    )

    mean_range = (float(scaler.mean_.min()), float(scaler.mean_.max()))
    std_range = (float(scaler.scale_.min()), float(scaler.scale_.max()))
    logger.debug(
        f"Scaler fitado — mean ∈ [{mean_range[0]:.4f}, {mean_range[1]:.4f}], "
        f"std ∈ [{std_range[0]:.6f}, {std_range[1]:.4f}]"
    )
    return scaler


def remap_labels(y: np.ndarray) -> np.ndarray:
    """Remapeia labels {-1, 0, 1} para {0, 1, 2} para CrossEntropyLoss do PyTorch.

    Mapeamento:
        SELL  : -1 → 0
        HOLD  :  0 → 1
        BUY   :  1 → 2

    Reversível: original = remapped - 1

    Args:
        y: Array com valores em {-1, 0, 1}, qualquer dtype inteiro.

    Returns:
        Array dtype np.int64 com valores em {0, 1, 2}.

    Raises:
        ValueError: Se y contiver valores fora de {-1, 0, 1}.
    """
    unique_vals = set(int(v) for v in np.unique(y))
    if not unique_vals.issubset({-1, 0, 1}):
        raise ValueError(
            f"[remap_labels] Labels devem ser subconjunto de {{-1, 0, 1}}. "
            f"Encontrado: {unique_vals}"
        )

    y_remapped: np.ndarray = (y.astype(np.int64)) + _LABEL_OFFSET

    unique_out = set(int(v) for v in np.unique(y_remapped))
    assert unique_out.issubset({0, 1, 2}), (
        f"[remap_labels] Output inválido após remapeamento: {unique_out}"
    )

    return y_remapped


def create_sliding_window_splits(
    parquet_path: Path | None = None,
    config: dict[str, Any] | None = None,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Pipeline completo: Parquet → arrays numpy de train/val/test escalados.

    Função de conveniência que encapsula o fluxo completo de transformação.
    Usada internamente por src/models/dataset.py.

    Args:
        parquet_path: Caminho para o Parquet (None usa default).
        config: Configuração do modelo (None carrega model_config.yaml).
        feature_cols: Colunas de features (None usa MODEL_FEATURE_COLS).

    Returns:
        Dicionário com chaves:
            - "X_train", "y_train": arrays de treino escalados e remapeados
            - "X_val",   "y_val":   arrays de validação escalados e remapeados
            - "X_test",  "y_test":  arrays de teste escalados e remapeados
            - "scaler":             StandardScaler fitado no train
            - "train_dates":        DatetimeIndex das datas finais de cada janela de treino
            - "val_dates":          DatetimeIndex das datas finais de cada janela de val
            - "test_dates":         DatetimeIndex das datas finais de cada janela de test
            - "feature_cols":       lista de features usadas
            - "window_size":        W usado
    """
    cfg = config or load_model_config()
    cols = feature_cols or MODEL_FEATURE_COLS

    window_size: int = cfg["model"]["window_size"]
    train_ratio: float = cfg["training"]["train_ratio"]
    val_ratio: float = cfg["training"]["val_ratio"]

    df = load_features_parquet(parquet_path)

    valid_indices = compute_valid_indices(df, window_size, cols)

    if len(valid_indices) == 0:
        raise ValueError(
            f"Nenhum índice válido encontrado para W={window_size}. "
            "O Parquet pode ser muito curto ou conter muitos NaNs."
        )

    train_idx, val_idx, test_idx = split_indices_chronological(
        valid_indices, train_ratio, val_ratio
    )

    X_train_raw, y_train_raw = build_windows(df, train_idx, window_size, cols)
    X_val_raw, y_val_raw = build_windows(df, val_idx, window_size, cols)
    X_test_raw, y_test_raw = build_windows(df, test_idx, window_size, cols)

    scaler = fit_scaler(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    y_train = remap_labels(y_train_raw)
    y_val = remap_labels(y_val_raw)
    y_test = remap_labels(y_test_raw)

    # Log distribuição de classes por split
    for split_name, y_split in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        unique, counts = np.unique(y_split, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts, strict=True)}
        logger.info(f"Distribuição de classes [{split_name}]: {dist}  (0=SELL, 1=HOLD, 2=BUY)")

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "scaler": scaler,
        "train_dates": df.index[train_idx],
        "val_dates": df.index[val_idx],
        "test_dates": df.index[test_idx],
        "feature_cols": cols,
        "window_size": window_size,
    }


# ---------------------------------------------------------------------------
# Entry point — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys

    print("=" * 70)
    print("Sliding Window — Smoke Test")
    print("=" * 70)

    result = create_sliding_window_splits()

    W = result["window_size"]
    F = len(result["feature_cols"])

    print(f"\nJanela: W={W} semanas × F={F} features → {W * F} neurônios de entrada")
    print("\nShapes:")
    for split in ("train", "val", "test"):
        X = result[f"X_{split}"]
        y = result[f"y_{split}"]
        dates = result[f"{split}_dates"]
        print(
            f"  {split.capitalize():5s}: X={X.shape}, y={y.shape}, "
            f"período: {dates[0].date()} → {dates[-1].date()}"
        )

    scaler: StandardScaler = result["scaler"]
    print("\nScaler (fitado apenas no train):")
    print(f"  mean ∈ [{scaler.mean_.min():.4f}, {scaler.mean_.max():.4f}]")
    print(f"  std  ∈ [{scaler.scale_.min():.6f}, {scaler.scale_.max():.4f}]")

    print("\nDistribuição de labels (0=SELL, 1=HOLD, 2=BUY):")
    for split in ("train", "val", "test"):
        y = result[f"y_{split}"]
        unique, counts = np.unique(y, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts, strict=True)}
        print(f"  {split.capitalize():5s}: {dist}")

    print("\n[OK] Smoke test concluído com sucesso.")
    sys.exit(0)
