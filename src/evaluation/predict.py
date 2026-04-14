"""
Inferência — Previsão para a semana atual com o modelo treinado.

Uso:
    uv run python -m src.evaluation.predict

Saída esperada:
    ╔══════════════════════════════════════════╗
    ║  AGRO3 — Sinal para 2025-04-11           ║
    ╠══════════════════════════════════════════╣
    ║  Decisão : VENDER                        ║
    ║  P(SELL) : 0.823   P(NOT-SELL) : 0.177  ║
    ║  Confiança: ALTA (> 70%)                 ║
    ╚══════════════════════════════════════════╝

O modelo usa a janela das últimas 52 semanas de features (preço, técnicos,
fundamentos, macro) carregadas do parquet já processado. O scaler é o mesmo
que foi fitado durante o treino — carregado do checkpoint mlp_v1.pt.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.feature_engineering.sliding_window import (
    MODEL_FEATURE_COLS,
    load_features_parquet,
)
from src.models.mlp import build_mlp_from_config
from src.models.trainer import Trainer
from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINT_PATH = _PROJECT_ROOT / "data" / "models" / "mlp_v1.pt"

_LABEL_MAP = {0: "VENDER", 1: "NÃO VENDER"}
_CONFIDENCE_THRESHOLDS = {"ALTA": 0.70, "MÉDIA": 0.55}


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


def predict_current(checkpoint_path: Path = _CHECKPOINT_PATH) -> dict[str, Any]:
    """Executa inferência na semana mais recente disponível no parquet.

    Carrega o scaler e o modelo do checkpoint, monta a janela das últimas
    W semanas e retorna a decisão com probabilidades.

    Args:
        checkpoint_path: Caminho para mlp_v1.pt.

    Returns:
        Dicionário com:
            date        : data da última semana usada como referência
            decision    : 'VENDER' ou 'NÃO VENDER'
            label       : 0 (SELL) ou 1 (NOT-SELL)
            p_sell      : probabilidade P(SELL)
            p_not_sell  : probabilidade P(NOT-SELL)
            confidence  : 'ALTA', 'MÉDIA' ou 'BAIXA'
            window_start: primeira semana da janela de 52 semanas
            window_end  : última semana da janela (= date)

    Raises:
        FileNotFoundError: Se mlp_v1.pt não existir.
        ValueError: Se o parquet não tiver semanas suficientes para uma janela.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint_path}. "
            "Execute 'uv run python -m src.models.trainer' primeiro."
        )

    device = _select_device()

    # --- 1. Carrega checkpoint (modelo + scaler fitado no treino) ---
    ckpt = Trainer.load_checkpoint(checkpoint_path, device)
    cfg: dict[str, Any] = ckpt["config"]
    scaler = ckpt["scaler"]
    window_size: int = cfg["model"]["window_size"]

    model = build_mlp_from_config(cfg, input_dim=ckpt["input_dim"])
    model.load_state_dict(ckpt["model_state_dict"])
    try:
        model = model.to(device)
    except RuntimeError:
        device = torch.device("cpu")
        model = model.to(device)
    model.eval()

    # --- 2. Carrega features e extrai a janela mais recente ---
    df = load_features_parquet()
    feature_df = df[MODEL_FEATURE_COLS].copy()

    # Remove linhas com NaN em qualquer feature (não podem entrar na janela)
    feature_df = feature_df.dropna()

    if len(feature_df) < window_size:
        raise ValueError(
            f"Apenas {len(feature_df)} semanas sem NaN disponíveis, "
            f"mas a janela exige {window_size}. "
            "Execute o pipeline de feature engineering."
        )

    window_df = feature_df.iloc[-window_size:]
    window_start: str = window_df.index[0].strftime("%Y-%m-%d")
    window_end: str = window_df.index[-1].strftime("%Y-%m-%d")

    # --- 3. Normaliza com o scaler do checkpoint (anti-leakage) ---
    X_raw = window_df.values.astype(np.float32)          # (W, F)
    X_flat = X_raw.reshape(1, -1)                         # (1, W*F)
    X_scaled = scaler.transform(X_flat).astype(np.float32)

    # --- 4. Forward pass ---
    with torch.no_grad():
        logits = model(torch.tensor(X_scaled, device=device))
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0]  # [P(SELL), P(NOT-SELL)]

    p_sell: float = float(proba[0])
    p_not_sell: float = float(proba[1])
    label: int = int(np.argmax(proba))  # 0=SELL, 1=NOT-SELL

    # --- 5. Nível de confiança ---
    max_p = max(p_sell, p_not_sell)
    if max_p >= _CONFIDENCE_THRESHOLDS["ALTA"]:
        confidence = "ALTA"
    elif max_p >= _CONFIDENCE_THRESHOLDS["MÉDIA"]:
        confidence = "MÉDIA"
    else:
        confidence = "BAIXA"

    return {
        "date": window_end,
        "decision": _LABEL_MAP[label],
        "label": label,
        "p_sell": p_sell,
        "p_not_sell": p_not_sell,
        "confidence": confidence,
        "window_start": window_start,
        "window_end": window_end,
    }


def print_prediction(result: dict[str, Any]) -> None:
    """Imprime o resultado de forma legível no terminal."""
    decision = result["decision"]
    p_sell = result["p_sell"]
    p_not_sell = result["p_not_sell"]
    confidence = result["confidence"]
    date = result["date"]
    w_start = result["window_start"]
    w_end = result["window_end"]

    # Ícone por decisão
    icon = "🔴 VENDER" if result["label"] == 0 else "🟢 NÃO VENDER"

    border = "=" * 50
    print(f"\n{border}")
    print(f"  AGRO3 — Sinal para {date}")
    print(border)
    print(f"  Decisão    : {icon}")
    print(f"  P(VENDER)  : {p_sell:.3f}   P(MANTER) : {p_not_sell:.3f}")
    print(f"  Confiança  : {confidence}")
    print(f"  Janela     : {w_start} → {w_end} ({result['window_end']})")
    print(border)

    # Aviso de baixa confiança
    if confidence == "BAIXA":
        print(
            "\n  ⚠️  Confiança BAIXA: probabilidades próximas de 50/50."
            "\n     Sinal inconclusivo — considere aguardar mais dados."
        )

    print()


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)  # suprime logs internos
    result = predict_current()
    print_prediction(result)
