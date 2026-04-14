"""
Threshold Calibration — Otimização pós-treinamento do limiar de decisão.

## Problema

O modelo v4 (Final Refit) usa `argmax(logit)` como critério de decisão,
equivalente a threshold=0.5 sobre P(SELL). Com distribuição de test
74% SELL / 26% NOT-SELL, o threshold neutro sub-detecta SELLs.

## Solução

Varrer t ∈ [0.10, 0.90] no conjunto de calibração (rf_val_idx: últimos 20%
do pré-test, ≈ 142 amostras de 2019-2021) e selecionar o t* que maximiza
F1-Macro. Aplicar t* no test set (2021-2025) sem re-exposição dos dados.

Predição final: SELL (0) se P(SELL) > t, senão NOT-SELL (1).

## Anti-Leakage

- O scaler é carregado DO CHECKPOINT (rf_scaler fitado em 2006-2019)
- O conjunto de calibração (rf_val_idx) é reconstruído deterministicamente
  pela mesma lógica do trainer — nunca foi usado no treinamento do refit
  (era apenas o monitor de early stopping)
- O test set permanece untouched durante a busca do threshold

## Uso

    uv run python -m src.evaluation.threshold_calibration
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from src.feature_engineering.sliding_window import (
    MODEL_FEATURE_COLS,
    build_windows,
    compute_valid_indices,
    compute_walk_forward_splits,
    load_features_parquet,
    remap_labels,
)
from src.models.dataset import AgRo3Dataset
from src.models.mlp import build_mlp_from_config
from src.models.trainer import Trainer
from src.utils.config import load_model_config
from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_PATH = _PROJECT_ROOT / "data" / "models" / "mlp_v1.pt"


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    try:
        if torch.backends.mps.is_available():
            torch.zeros(1, device="mps")
            return torch.device("mps")
    except RuntimeError:
        pass
    return torch.device("cpu")


def _get_probabilities(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Executa forward pass e retorna P(SELL) para cada amostra.

    Args:
        model: Modelo MLP em modo eval.
        X: Features normalizadas, shape (N, input_dim).
        device: Dispositivo PyTorch.
        batch_size: Tamanho do batch para inferência.

    Returns:
        Array de shape (N,) com P(classe 0 = SELL) para cada amostra.
    """
    model.eval()
    all_proba: list[np.ndarray] = []

    # Dummy dataset só para usar DataLoader
    y_dummy = np.zeros(len(X), dtype=np.int64)
    ds = AgRo3Dataset(X, y_dummy)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for X_batch, _ in loader:
            logits = model(X_batch.to(device))
            proba = torch.softmax(logits, dim=1)
            all_proba.append(proba[:, 0].cpu().numpy())  # P(SELL=0)

    return np.concatenate(all_proba)


def _apply_threshold(proba_sell: np.ndarray, threshold: float) -> np.ndarray:
    """Aplica threshold: prediz SELL (0) se P(SELL) > threshold, senão NOT-SELL (1).

    Args:
        proba_sell: Array P(SELL) para cada amostra, shape (N,).
        threshold: Limiar de decisão em (0, 1).

    Returns:
        Predições binárias em {0, 1}.
    """
    return np.where(proba_sell > threshold, 0, 1).astype(np.int64)


def _metrics_at_threshold(
    y_true: np.ndarray,
    proba_sell: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Computa métricas-chave para um dado threshold.

    Args:
        y_true: Labels reais em {0, 1}.
        proba_sell: P(SELL) para cada amostra.
        threshold: Limiar de decisão.

    Returns:
        Dicionário com f1_macro, f1_sell, f1_not_sell, precision_sell,
        recall_sell, precision_not_sell, recall_not_sell.
    """
    y_pred = _apply_threshold(proba_sell, threshold)

    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_sell": float(
            f1_score(y_true, y_pred, labels=[0], average=None, zero_division=0)[0]
        ),
        "f1_not_sell": float(
            f1_score(y_true, y_pred, labels=[1], average=None, zero_division=0)[0]
        ),
        "precision_sell": float(
            precision_score(y_true, y_pred, labels=[0], average=None, zero_division=0)[0]
        ),
        "recall_sell": float(
            recall_score(y_true, y_pred, labels=[0], average=None, zero_division=0)[0]
        ),
        "precision_not_sell": float(
            precision_score(y_true, y_pred, labels=[1], average=None, zero_division=0)[0]
        ),
        "recall_not_sell": float(
            recall_score(y_true, y_pred, labels=[1], average=None, zero_division=0)[0]
        ),
    }


def calibrate_threshold(
    checkpoint_path: Path = _CHECKPOINT_PATH,
    n_thresholds: int = 81,
) -> dict[str, Any]:
    """Encontra o threshold ótimo por varredura no conjunto de calibração.

    O conjunto de calibração é o rf_val_idx (últimos 20% do pré-test),
    reconstruído deterministicamente — mesma lógica do Final Refit no trainer.

    Args:
        checkpoint_path: Caminho para mlp_v1.pt (contém rf_scaler).
        n_thresholds: Número de thresholds a testar em [0.10, 0.90].

    Returns:
        Dicionário com:
            optimal_threshold: t* que maximiza F1-Macro no val de calibração.
            val_metrics_at_default: Métricas do val com t=0.5.
            val_metrics_at_optimal: Métricas do val com t=t*.
            test_metrics_at_default: Métricas do test com t=0.5.
            test_metrics_at_optimal: Métricas do test com t=t*.
            threshold_sweep: Lista de (threshold, f1_macro_val) para plotagem.
            y_true_test: Labels reais do test (para análise).
            y_pred_test_optimal: Predições com t* no test.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint_path}. "
            f"Execute 'uv run python -m src.models.trainer' primeiro."
        )

    device = _select_device()
    logger.info(f"Device: {device}")

    # --- 1. Carrega checkpoint (inclui rf_scaler) ---
    ckpt = Trainer.load_checkpoint(checkpoint_path, device)
    cfg: dict[str, Any] = ckpt["config"]
    scaler = ckpt["scaler"]  # rf_scaler fitado em 2006-2019

    model = build_mlp_from_config(cfg, input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    try:
        model = model.to(device)
    except RuntimeError:
        device = torch.device("cpu")
        model = model.to(device)
    model.eval()

    logger.info(
        f"Modelo carregado: best_epoch={ckpt['best_epoch']} | "
        f"best_val_f1={ckpt['best_val_f1']:.4f}"
    )

    # --- 2. Reconstrói splits (mesma lógica do _run_walk_forward) ---
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    window_size: int = model_cfg["window_size"]
    n_splits: int = train_cfg["walk_forward_n_splits"]
    test_ratio: float = train_cfg["test_ratio"]
    min_train: int = train_cfg["walk_forward_min_train"]
    feature_cols = MODEL_FEATURE_COLS

    df = load_features_parquet()
    valid_indices = compute_valid_indices(df, window_size, feature_cols)
    test_indices, _ = compute_walk_forward_splits(
        valid_indices, n_splits, test_ratio, min_train
    )

    n_pre_test = len(valid_indices) - len(test_indices)
    pre_test_indices = valid_indices[:n_pre_test]
    n_rf_val = max(50, int(n_pre_test * 0.20))
    rf_val_idx = pre_test_indices[-n_rf_val:]

    logger.info(
        f"Conjunto de calibração: {len(rf_val_idx)} amostras (rf_val_idx) | "
        f"Test: {len(test_indices)} amostras"
    )

    # --- 3. Constrói features dos dois conjuntos com o scaler do checkpoint ---
    X_cal_raw, y_cal_raw = build_windows(df, rf_val_idx, window_size, feature_cols)
    X_test_raw, y_test_raw = build_windows(df, test_indices, window_size, feature_cols)

    X_cal = scaler.transform(X_cal_raw)
    X_test = scaler.transform(X_test_raw)

    y_cal = remap_labels(y_cal_raw)
    y_test = remap_labels(y_test_raw)

    logger.info(
        f"Cal labels: SELL={np.sum(y_cal==0)} NOT-SELL={np.sum(y_cal==1)} | "
        f"Test labels: SELL={np.sum(y_test==0)} NOT-SELL={np.sum(y_test==1)}"
    )

    # --- 4. Probabilidades em ambos os conjuntos ---
    proba_sell_cal = _get_probabilities(model, X_cal, device)
    proba_sell_test = _get_probabilities(model, X_test, device)

    # --- 5. Varredura de thresholds no conjunto de calibração ---
    thresholds = np.linspace(0.10, 0.90, n_thresholds)
    sweep: list[tuple[float, float]] = []

    for t in thresholds:
        m = _metrics_at_threshold(y_cal, proba_sell_cal, t)
        sweep.append((float(t), m["f1_macro"]))

    best_t, best_f1_cal = max(sweep, key=lambda x: x[1])

    logger.info(
        f"Threshold ótimo (cal): t*={best_t:.2f} → F1-Macro(cal)={best_f1_cal:.4f} "
        f"[vs t=0.50 → F1-Macro(cal)="
        f"{_metrics_at_threshold(y_cal, proba_sell_cal, 0.50)['f1_macro']:.4f}]"
    )

    # --- 6. Métricas no val (calibração) ---
    val_default = _metrics_at_threshold(y_cal, proba_sell_cal, 0.50)
    val_optimal = _metrics_at_threshold(y_cal, proba_sell_cal, best_t)

    # --- 7. Métricas no test (ground truth) ---
    test_default = _metrics_at_threshold(y_test, proba_sell_test, 0.50)
    test_optimal = _metrics_at_threshold(y_test, proba_sell_test, best_t)

    # --- 8. Confusion matrices ---
    y_pred_test_default = _apply_threshold(proba_sell_test, 0.50)
    y_pred_test_optimal = _apply_threshold(proba_sell_test, best_t)

    cm_default = confusion_matrix(y_test, y_pred_test_default, labels=[0, 1])
    cm_optimal = confusion_matrix(y_test, y_pred_test_optimal, labels=[0, 1])

    return {
        "optimal_threshold": best_t,
        "val_metrics_at_default": val_default,
        "val_metrics_at_optimal": val_optimal,
        "test_metrics_at_default": test_default,
        "test_metrics_at_optimal": test_optimal,
        "confusion_matrix_default": cm_default,
        "confusion_matrix_optimal": cm_optimal,
        "threshold_sweep": sweep,
        "y_true_test": y_test,
        "y_pred_test_optimal": y_pred_test_optimal,
        "n_cal": len(y_cal),
        "n_test": len(y_test),
    }


def print_calibration_report(results: dict[str, Any]) -> None:
    """Imprime relatório comparativo antes/depois da calibração."""
    sep = "=" * 65

    print(f"\n{sep}")
    print("  THRESHOLD CALIBRATION REPORT — AGRO3 Binary Classifier v4")
    print(sep)
    print(f"  Conjunto de calibração: N={results['n_cal']} (rf_val 2019-2021)")
    print(f"  Conjunto de teste:      N={results['n_test']} (test 2021-2025)")
    print(f"  Threshold ótimo (t*):   {results['optimal_threshold']:.2f}  "
          f"[padrão: 0.50]")
    print(sep)

    def _row(label: str, d: float, o: float, higher_better: bool = True) -> str:
        arrow = "▲" if (o > d) == higher_better else ("▼" if o != d else " ")
        return f"  {label:<28s}  {d:.4f}  →  {o:.4f}  {arrow}"

    print("\n  --- CONJUNTO DE CALIBRAÇÃO (val 2019-2021) ---")
    print(f"  {'Métrica':<28s}  {'t=0.50':>6s}     {'t=t*':>6s}")
    print(f"  {'-'*55}")
    d, o = results["val_metrics_at_default"], results["val_metrics_at_optimal"]
    for key, label in [
        ("f1_macro", "F1-Macro"),
        ("f1_sell", "F1 (SELL)"),
        ("f1_not_sell", "F1 (NOT-SELL)"),
        ("precision_sell", "Precision (SELL)"),
        ("recall_sell", "Recall (SELL)"),
        ("precision_not_sell", "Precision (NOT-SELL)"),
        ("recall_not_sell", "Recall (NOT-SELL)"),
    ]:
        print(_row(label, d[key], o[key]))

    print(f"\n  --- TEST SET (2021-2025) — resultado final ---")
    print(f"  {'Métrica':<28s}  {'t=0.50':>6s}     {'t=t*':>6s}")
    print(f"  {'-'*55}")
    d, o = results["test_metrics_at_default"], results["test_metrics_at_optimal"]
    for key, label in [
        ("f1_macro", "F1-Macro  ← métrica primária"),
        ("f1_sell", "F1 (SELL)"),
        ("f1_not_sell", "F1 (NOT-SELL)"),
        ("precision_sell", "Precision (SELL)"),
        ("recall_sell", "Recall (SELL)"),
        ("precision_not_sell", "Precision (NOT-SELL)"),
        ("recall_not_sell", "Recall (NOT-SELL)"),
    ]:
        print(_row(label, d[key], o[key]))

    cm_d = results["confusion_matrix_default"]
    cm_o = results["confusion_matrix_optimal"]
    print(f"\n  --- CONFUSION MATRIX ---")
    print(f"  Pred →          SELL    NOT-SELL")
    print(f"  t=0.50  SELL    {cm_d[0,0]:5d}    {cm_d[0,1]:5d}  (support={cm_d[0].sum()})")
    print(f"  t=0.50  NOTSELL {cm_d[1,0]:5d}    {cm_d[1,1]:5d}  (support={cm_d[1].sum()})")
    print()
    print(f"  t={results['optimal_threshold']:.2f}  SELL    {cm_o[0,0]:5d}    {cm_o[0,1]:5d}  (support={cm_o[0].sum()})")
    print(f"  t={results['optimal_threshold']:.2f}  NOTSELL {cm_o[1,0]:5d}    {cm_o[1,1]:5d}  (support={cm_o[1].sum()})")

    print(f"\n{sep}\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = calibrate_threshold()
    print_calibration_report(results)
