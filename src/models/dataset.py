"""
AgRo3Dataset — Fase 3: Wrapper PyTorch para os dados de sliding window.

## Responsabilidade

Este módulo é a camada PyTorch do pipeline de Fase 3. Ele consome os arrays
numpy gerados por `src/feature_engineering/sliding_window.py` e os encapsula
em um `torch.utils.data.Dataset` compatível com `DataLoader`.

## Label Encoding

PyTorch's CrossEntropyLoss requer targets em [0, num_classes). O mapeamento é:
    SELL  : -1 → 0
    HOLD  :  0 → 1
    BUY   :  1 → 2

Para reverter: original_label = tensor_label - 1

## Uso Recomendado

    from src.models.dataset import create_datasets
    from torch.utils.data import DataLoader

    train_ds, val_ds, test_ds, scaler = create_datasets()

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    # Persista o scaler junto ao checkpoint do modelo:
    import joblib
    joblib.dump(scaler, "data/models/scaler.pkl")

## Anti-Leakage

O `StandardScaler` é fitado exclusivamente nos dados de treino dentro de
`create_datasets()`. Val e test recebem o transform com os parâmetros do
treino. Este invariante é crítico: nunca refit o scaler em produção.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from src.feature_engineering.sliding_window import (
    MODEL_FEATURE_COLS,
    create_sliding_window_splits,
)
from src.utils.config import load_model_config
from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)


class AgRo3Dataset(Dataset):  # type: ignore[type-arg]
    """Dataset PyTorch para o classificador MLP da AGRO3.

    Encapsula arrays numpy (já escalados e com labels remapeados) como
    Tensors PyTorch. Toda a lógica de transformação (sliding window,
    scaler, remapeamento) ocorre fora desta classe, em sliding_window.py.

    Atributos:
        X: Tensor float32 de shape (N, W*F) — entrada da MLP.
        y: Tensor int64 de shape (N,) — labels em {0, 1, 2}.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Args:
            X: Array numpy (N, W*F) float64 — será convertido para float32.
            y: Array numpy (N,) int64 com labels em {0, 1, 2}.
        """
        assert X.ndim == 2, f"X deve ser 2D, recebido shape: {X.shape}"
        assert y.ndim == 1, f"y deve ser 1D, recebido shape: {y.shape}"
        assert len(X) == len(y), (
            f"X e y devem ter o mesmo número de amostras. "
            f"X: {len(X)}, y: {len(y)}"
        )

        # float32 para eficiência em GPU; int64 obrigatório para CrossEntropyLoss
        self.X: torch.Tensor = torch.from_numpy(X.astype(np.float32))
        self.y: torch.Tensor = torch.from_numpy(y.astype(np.int64))

        logger.debug(
            f"AgRo3Dataset criado: {len(self)} amostras, "
            f"input_dim={self.X.shape[1]}"
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retorna (X[idx], y[idx]) como Tensors PyTorch.

        A conversão numpy→Tensor já ocorreu no __init__. Este método
        é O(1) e não realiza nenhuma transformação adicional.
        """
        return self.X[idx], self.y[idx]

    @property
    def input_dim(self) -> int:
        """Dimensão do vetor de entrada (W * F)."""
        return int(self.X.shape[1])

    @property
    def num_classes(self) -> int:
        """Número de classes únicas no dataset."""
        return int(self.y.unique().numel())


def create_datasets(
    parquet_path: Path | None = None,
    config: dict[str, Any] | None = None,
    feature_cols: list[str] | None = None,
) -> tuple[AgRo3Dataset, AgRo3Dataset, AgRo3Dataset, StandardScaler]:
    """Cria os três Datasets PyTorch a partir do Parquet da Fase 2.

    Este é o entry point recomendado para o Trainer da Fase 4. Executa
    o pipeline completo:
        1. Carrega o Parquet
        2. Computa índices válidos (sem NaN, com target)
        3. Divide cronologicamente (train/val/test)
        4. Constrói janelas deslizantes (W, F) → (W*F,) por amostra
        5. Fita StandardScaler APENAS no train
        6. Escala todos os splits com parâmetros do train
        7. Remapeia labels {-1,0,1} → {0,1,2}
        8. Envolve em AgRo3Dataset

    Args:
        parquet_path: Caminho para o Parquet. None usa o default da Fase 2.
        config: Configuração do modelo. None carrega model_config.yaml.
        feature_cols: Colunas de features. None usa MODEL_FEATURE_COLS (23).

    Returns:
        Tupla (train_ds, val_ds, test_ds, scaler):
            - train_ds: AgRo3Dataset de treino
            - val_ds:   AgRo3Dataset de validação
            - test_ds:  AgRo3Dataset de teste
            - scaler:   StandardScaler fitado no train — persista com o modelo

    Raises:
        FileNotFoundError: Se o Parquet não existir.
        ValueError: Se não houver amostras válidas suficientes para os splits.
    """
    cfg = config or load_model_config()

    logger.info("=== create_datasets: iniciando pipeline Pandas → PyTorch ===")

    splits = create_sliding_window_splits(
        parquet_path=parquet_path,
        config=cfg,
        feature_cols=feature_cols or MODEL_FEATURE_COLS,
    )

    train_ds = AgRo3Dataset(splits["X_train"], splits["y_train"])
    val_ds = AgRo3Dataset(splits["X_val"], splits["y_val"])
    test_ds = AgRo3Dataset(splits["X_test"], splits["y_test"])
    scaler: StandardScaler = splits["scaler"]

    logger.info(
        f"Datasets criados — "
        f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)} | "
        f"input_dim: {train_ds.input_dim}"
    )

    return train_ds, val_ds, test_ds, scaler


# ---------------------------------------------------------------------------
# Entry point — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys

    print("=" * 70)
    print("AgRo3Dataset — Smoke Test")
    print("=" * 70)

    train_ds, val_ds, test_ds, scaler = create_datasets()

    print(f"\nInput dim: {train_ds.input_dim} neurônios")
    print("\nDatasets:")
    print(f"  Train: {len(train_ds):4d} amostras | classes: {train_ds.num_classes}")
    print(f"  Val:   {len(val_ds):4d} amostras | classes: {val_ds.num_classes}")
    print(f"  Test:  {len(test_ds):4d} amostras | classes: {test_ds.num_classes}")

    x0, y0 = train_ds[0]
    print("\nAmostra [0] do Train:")
    print(f"  X.shape: {x0.shape}, dtype: {x0.dtype}")
    print(f"  y:       {y0.item()} (0=SELL, 1=HOLD, 2=BUY)")
    print(f"  X[:5]:   {x0[:5].tolist()}")

    # Verifica compatibilidade com DataLoader
    from torch.utils.data import DataLoader

    loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    X_batch, y_batch = next(iter(loader))
    print("\nBatch de DataLoader:")
    print(f"  X_batch.shape: {X_batch.shape}")
    print(f"  y_batch.shape: {y_batch.shape}")
    print(f"  y_batch:       {y_batch.tolist()}")

    print("\n[OK] Smoke test concluído com sucesso.")
    sys.exit(0)
