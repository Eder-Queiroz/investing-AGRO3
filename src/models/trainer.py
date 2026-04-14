"""
Trainer — Fase 4: Loop de treinamento para o classificador MLP da AGRO3.

## Responsabilidade

Orquestra todo o ciclo de treinamento da MLP:
  - Critério de loss com balanceamento de classes e label smoothing
  - Otimizador AdamW com cosine annealing
  - Clipping de gradientes para estabilidade em dados financeiros
  - Early stopping monitorando val F1-Macro (não acurácia)
  - Checkpoint com model state + scaler (artefato auto-suficiente para inferência)
  - Métricas per-class (SELL/HOLD/BUY) para detectar colapso de classes

## Decisões de Design Chave

**Early stopping em F1-Macro, não acurácia**: Com HOLD dominante, um modelo
que prevê somente HOLD atinge ~60-70% de acurácia. F1-Macro penaliza igualmente
o fracasso em qualquer classe.

**Label smoothing = 0.05**: O threshold CDI±5pp é inerentemente ruidoso — uma
amostra em CDI+4.9pp é HOLD quando o sinal real é indistinguível de BUY.
Smoothing modela essa incerteza. 0.05 (não 0.1) evita suavizar demais o sinal.

**Gradient clipping max_norm=1.0**: Superfícies de loss financeiras têm spikes.
Um único batch de crise de mercado pode explodir gradientes, desfazendo 10
épocas de aprendizado. Clipping previne passos destrutivos.

**class_weights via loss, não WeightedRandomSampler**: Pesos no loss corrigem
o gradiente sem distorcer a distribuição de sampling do treinamento.

**zero_division=0 no F1**: Se o modelo colapsar para uma classe nas primeiras
épocas, F1 undefined → NaN corromperia o critério de early stopping silenciosamente.

## Uso

    uv run python -m src.models.trainer

## Checkpoint (data/models/mlp_v1.pt)

O checkpoint inclui tudo necessário para inferência offline:
    model_state_dict, optimizer_state_dict, config, input_dim, scaler,
    best_epoch, best_val_f1, metrics_history, class_weights
"""

from __future__ import annotations

import dataclasses
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.feature_engineering.sliding_window import (
    MODEL_FEATURE_COLS,
    build_windows,
    compute_valid_indices,
    compute_walk_forward_splits,
    fit_scaler,
    load_features_parquet,
    remap_labels,
)
from src.models.dataset import AgRo3Dataset, create_datasets
from src.models.mlp import build_mlp_from_config
from src.utils.config import load_model_config
from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# Mapeamento de índice → rótulo legível (binário: SELL vs. NOT-SELL)
_CLASS_NAMES: dict[int, str] = {0: "SELL", 1: "NOT-SELL"}


# ---------------------------------------------------------------------------
# MetricsBundle
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MetricsBundle:
    """Métricas de uma única época de avaliação.

    Serializa via dataclasses.asdict() para inclusão no checkpoint.
    """

    epoch: int
    train_loss: float
    val_loss: float
    val_acc: float
    val_f1_macro: float  # métrica primária de early stopping
    val_f1_sell: float
    val_f1_not_sell: float

    def log_summary(self) -> None:
        """Emite uma linha de log INFO com o resumo da época."""
        logger.info(
            f"Epoch {self.epoch:03d} | "
            f"train_loss={self.train_loss:.4f} | "
            f"val_loss={self.val_loss:.4f} | "
            f"val_acc={self.val_acc:.3f} | "
            f"f1_macro={self.val_f1_macro:.4f} | "
            f"f1[SELL={self.val_f1_sell:.3f} "
            f"NOT-SELL={self.val_f1_not_sell:.3f}]"
        )


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------


class EarlyStopping:
    """Critério de parada antecipada monitorando uma métrica escalar.

    Attributes:
        should_stop: True quando patience foi esgotado.
        best_score: Melhor valor observado até o momento.
        best_epoch: Época onde o melhor valor foi atingido.
    """

    def __init__(
        self,
        patience: int,
        min_delta: float = 0.001,
        mode: str = "max",
    ) -> None:
        """
        Args:
            patience: Épocas a aguardar sem melhora antes de parar.
            min_delta: Melhora mínima para contar como progresso.
            mode: 'max' para métricas onde maior é melhor (F1), 'min' para loss.

        Raises:
            ValueError: Se mode não for 'max' ou 'min'.
        """
        if mode not in ("max", "min"):
            raise ValueError(f"mode deve ser 'max' ou 'min'. Recebido: '{mode}'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.should_stop: bool = False
        self.best_score: float = float("-inf") if mode == "max" else float("inf")
        self.best_epoch: int = 0
        self._counter: int = 0

    def step(self, score: float, epoch: int) -> bool:
        """Atualiza o estado com o score da época atual.

        Args:
            score: Valor da métrica monitorada nesta época.
            epoch: Índice da época atual (0-based).

        Returns:
            True se o treinamento deve ser interrompido.
        """
        improved = (
            score > self.best_score + self.min_delta
            if self.mode == "max"
            else score < self.best_score - self.min_delta
        )

        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping ativado na época {epoch}. "
                    f"Melhor {self.mode}={self.best_score:.4f} na época {self.best_epoch}."
                )

        return self.should_stop


# ---------------------------------------------------------------------------
# compute_class_weights
# ---------------------------------------------------------------------------


def compute_class_weights(
    y_train: np.ndarray,
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    """Computa pesos balanceados por classe para CrossEntropyLoss.

    Fórmula: weight[c] = n_samples / (num_classes * count[c])
    Equivalente a sklearn's compute_class_weight('balanced').

    Args:
        y_train: Array 1D de inteiros com labels em {0, 1, ..., num_classes-1}.
        num_classes: Total de classes (2 para SELL/NOT-SELL).
        device: Dispositivo alvo do tensor retornado.

    Returns:
        Tensor float32 de shape (num_classes,) no device especificado.

    Raises:
        ValueError: Se alguma classe estiver ausente de y_train (divisão por zero).
    """
    classes = np.arange(num_classes)
    present_classes = np.unique(y_train)

    if len(present_classes) < num_classes:
        missing = set(range(num_classes)) - set(present_classes.tolist())
        raise ValueError(
            f"Classes ausentes em y_train: {missing}. "
            f"Não é possível computar pesos — divisão por zero."
        )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    weight_info = {_CLASS_NAMES[i]: f"{w:.4f}" for i, w in enumerate(weights)}
    logger.info(f"Class weights computados: {weight_info}")

    return torch.tensor(weights, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class Trainer:
    """Orquestrador do ciclo de treinamento da MLP.

    Responsabilidades:
      - Executar epochs de treino e validação
      - Monitorar val F1-Macro com early stopping
      - Salvar checkpoint apenas quando val F1-Macro melhora
      - Logar métricas e confusion matrix em eventos-chave

    Não é responsável por: criar datasets, DataLoaders ou o modelo.
    """

    def __init__(
        self,
        model: nn.Module,
        config: dict[str, Any],
        device: torch.device,
        class_weights: torch.Tensor,
        scaler: StandardScaler,
    ) -> None:
        """
        Args:
            model: Modelo nn.Module (DIP — não acoplado a ValueInvestingMLP).
            config: Configuração completa do model_config.yaml.
            device: Dispositivo onde o modelo e dados residem.
            class_weights: Tensor (num_classes,) com pesos para CrossEntropyLoss.
            scaler: StandardScaler fitado no train — persistido no checkpoint.
        """
        self.model = model
        self.device = device
        self.scaler = scaler
        self.history: list[MetricsBundle] = []

        train_cfg: dict[str, Any] = config.get("training", config)

        self.epochs: int = train_cfg["epochs"]
        self.config = config

        # Otimizador
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

        # Loss com balanceamento e label smoothing
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=0.05,
        )

        # Scheduler: CosineAnnealingLR com T_max = epochs (decaimento suave único)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
        )

        # Early stopping monitorando val_f1_macro (maior é melhor)
        self.early_stopping = EarlyStopping(
            patience=train_cfg["early_stopping_patience"],
            min_delta=0.001,
            mode="max",
        )

    # ------------------------------------------------------------------
    # Loop de treino
    # ------------------------------------------------------------------

    def _train_epoch(
        self,
        train_loader: DataLoader,  # type: ignore[type-arg]
    ) -> float:
        """Executa uma época de treinamento.

        Args:
            train_loader: DataLoader do split de treino.

        Returns:
            Loss médio da época.
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()

            # Clipping de gradientes: previne passos destrutivos em spikes financeiros
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _val_epoch(
        self,
        val_loader: DataLoader,  # type: ignore[type-arg]
        epoch: int,
        train_loss: float,
    ) -> MetricsBundle:
        """Executa uma época de validação e computa métricas.

        Todas as métricas (F1, acurácia) são computadas em numpy/sklearn
        após acumular predições no CPU — nunca em tensors mid-loop.

        Args:
            val_loader: DataLoader do split de validação (ou teste).
            epoch: Índice da época atual (para o MetricsBundle).
            train_loss: Loss de treino desta época.

        Returns:
            MetricsBundle com todas as métricas da época.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)
                preds = logits.argmax(dim=1)

                total_loss += loss.item()
                n_batches += 1
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.cpu().numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        val_loss = total_loss / max(n_batches, 1)

        # zero_division=0: previne NaN se o modelo colapsar para uma classe
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        val_acc = float(accuracy_score(y_true, y_pred))

        return MetricsBundle(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            val_f1_macro=f1_macro,
            val_f1_sell=float(f1_per_class[0]),
            val_f1_not_sell=float(f1_per_class[1]),
        )

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        path: Path,
        metrics: MetricsBundle,
    ) -> None:
        """Salva checkpoint completo com model state + scaler.

        O checkpoint é auto-suficiente: contém tudo necessário para
        reconstruir o modelo e fazer inferência sem config externo.

        Args:
            path: Caminho destino do arquivo .pt.
            metrics: MetricsBundle da melhor época.
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_epoch": metrics.epoch,
            "best_val_f1": metrics.val_f1_macro,
            "config": self.config,
            "input_dim": _get_input_dim(self.model),
            "scaler": self.scaler,
            "metrics_history": [dataclasses.asdict(m) for m in self.history],
            "class_weights": self.criterion.weight.cpu().tolist()
            if self.criterion.weight is not None
            else None,
        }

        torch.save(checkpoint, path)
        logger.info(
            f"Checkpoint salvo: {path} "
            f"(época {metrics.epoch}, val_f1_macro={metrics.val_f1_macro:.4f})"
        )

    @staticmethod
    def load_checkpoint(
        path: Path,
        device: torch.device,
    ) -> dict[str, Any]:
        """Carrega checkpoint de disco.

        Args:
            path: Caminho do arquivo .pt.
            device: Dispositivo para mapeamento dos tensors.

        Returns:
            Dicionário com todos os campos do checkpoint.

        Raises:
            FileNotFoundError: Se o arquivo não existir.
            RuntimeError: Se o arquivo estiver corrompido.
        """
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {path}")

        try:
            checkpoint = torch.load(
                path,
                map_location=device,
                weights_only=False,  # necessário para deserializar o scaler sklearn
            )
        except Exception as exc:
            raise RuntimeError(f"Falha ao carregar checkpoint {path}: {exc}") from exc

        logger.info(
            f"Checkpoint carregado: {path} "
            f"(melhor época {checkpoint.get('best_epoch')}, "
            f"val_f1={checkpoint.get('best_val_f1', 'N/A'):.4f})"
        )
        return checkpoint

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,  # type: ignore[type-arg]
        val_loader: DataLoader,  # type: ignore[type-arg]
        checkpoint_path: Path,
    ) -> MetricsBundle:
        """Executa o treinamento completo com early stopping.

        Salva checkpoint apenas quando val_f1_macro melhora.
        Loga confusion matrix em: novo melhor checkpoint, early stopping, época final.

        Args:
            train_loader: DataLoader do split de treino (shuffle=True).
            val_loader: DataLoader do split de validação (shuffle=False).
            checkpoint_path: Caminho destino para o melhor checkpoint.

        Returns:
            MetricsBundle da melhor época observada.
        """
        logger.info(
            f"=== Iniciando treinamento: {self.epochs} épocas máximas | "
            f"device={self.device} | checkpoint={checkpoint_path} ==="
        )

        best_metrics: MetricsBundle | None = None

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(train_loader)
            metrics = self._val_epoch(val_loader, epoch=epoch, train_loss=train_loss)
            self.history.append(metrics)
            metrics.log_summary()

            self.scheduler.step()

            # Salva checkpoint quando val_f1_macro melhora
            if best_metrics is None or metrics.val_f1_macro > best_metrics.val_f1_macro:
                best_metrics = metrics
                self.save_checkpoint(checkpoint_path, metrics)
                self._log_confusion_matrix(val_loader, label="val (novo melhor)")

            # Early stopping
            stop = self.early_stopping.step(metrics.val_f1_macro, epoch=epoch)
            if stop:
                self._log_confusion_matrix(val_loader, label="val (early stopping)")
                break

        # Confusion matrix na época final (se não foi early stopping)
        if not self.early_stopping.should_stop and best_metrics is not None:
            self._log_confusion_matrix(val_loader, label="val (época final)")

        assert best_metrics is not None, "Nenhuma época foi executada."
        logger.info(
            f"=== Treinamento concluído. "
            f"Melhor val_f1_macro={best_metrics.val_f1_macro:.4f} "
            f"na época {best_metrics.epoch} ==="
        )
        return best_metrics

    # ------------------------------------------------------------------
    # Utilitários internos
    # ------------------------------------------------------------------

    def _log_confusion_matrix(
        self,
        loader: DataLoader,  # type: ignore[type-arg]
        label: str = "val",
    ) -> None:
        """Computa e loga a confusion matrix para o loader fornecido."""
        self.model.eval()
        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                logits = self.model(X_batch.to(self.device))
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_targets)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        logger.info(f"Confusion Matrix [{label}]:")
        logger.info("  Classes: SELL(0), NOT-SELL(1)")
        logger.info("  Pred →    SELL  NOT-SELL")
        for i, row in enumerate(cm):
            logger.info(f"  {_CLASS_NAMES[i]:10s} ↓  {row[0]:5d}  {row[1]:5d}")


def _get_input_dim(model: nn.Module) -> int:
    """Extrai input_dim do primeiro Linear do modelo."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            return module.in_features
    return -1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _verify_mps(model: nn.Module, input_dim: int, device: torch.device) -> torch.device:
    """Verifica compatibilidade MPS; faz fallback para CPU se necessário."""
    if device.type != "mps":
        return device
    try:
        probe = torch.randn(2, input_dim, device=device)
        model.to(device)
        model(probe)
        logger.info("MPS: forward pass de verificação OK.")
        return device
    except RuntimeError as exc:
        logger.warning(f"MPS indisponível ({exc}). Fallback para CPU.")
        model.cpu()
        return torch.device("cpu")


def _evaluate_on_test(
    trainer: "Trainer",
    model: nn.Module,
    checkpoint_path: Path,
    test_loader: "DataLoader[Any]",  # type: ignore[type-arg]
    best_epoch: int,
    device: torch.device,
) -> None:
    """Carrega melhor checkpoint e avalia no test set."""
    logger.info("=== Avaliação no Test Set (melhor checkpoint) ===")
    ckpt = Trainer.load_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_metrics = trainer._val_epoch(test_loader, epoch=best_epoch, train_loss=0.0)
    trainer._log_confusion_matrix(test_loader, label="test (final)")
    logger.info(
        f"TEST FINAL | "
        f"loss={test_metrics.val_loss:.4f} | "
        f"acc={test_metrics.val_acc:.3f} | "
        f"f1_macro={test_metrics.val_f1_macro:.4f} | "
        f"f1[SELL={test_metrics.val_f1_sell:.3f} "
        f"NOT-SELL={test_metrics.val_f1_not_sell:.3f}]"
    )


def _run_single_split(
    cfg: dict[str, Any],
    device: torch.device,
    checkpoint_path: Path,
) -> None:
    """Treinamento com split cronológico único (modo legado 60/20/20)."""
    train_ds, val_ds, test_ds, scaler = create_datasets(config=cfg)

    class_weights = compute_class_weights(
        y_train=train_ds.y.numpy(),
        num_classes=cfg["model"]["num_classes"],
        device=device,
    )

    batch_size: int = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = build_mlp_from_config(cfg, input_dim=train_ds.input_dim)
    device = _verify_mps(model, train_ds.input_dim, device)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Modelo pronto: {n_params:,} parâmetros | input_dim={train_ds.input_dim} | "
        f"train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}"
    )

    trainer = Trainer(
        model=model, config=cfg, device=device,
        class_weights=class_weights, scaler=scaler,
    )
    best_metrics = trainer.fit(train_loader, val_loader, checkpoint_path)
    _evaluate_on_test(trainer, model, checkpoint_path, test_loader, best_metrics.epoch, device)
    logger.info(f"Artefato salvo em: {checkpoint_path.resolve()}")


def _run_walk_forward(
    cfg: dict[str, Any],
    device: torch.device,
    checkpoint_path: Path,
) -> None:
    """Treinamento com walk-forward cross-validation (N folds expansivos).

    Cada fold treina um modelo do zero sobre um conjunto de dados diferente,
    garantindo que o modelo veja múltiplos regimes de mercado durante a
    validação. O checkpoint final é o da fold com maior val_f1_macro.
    O test set (últimos test_ratio% dos dados) é fixo e nunca visto durante CV.
    """
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    batch_size: int = train_cfg["batch_size"]
    n_splits: int = train_cfg["walk_forward_n_splits"]
    min_train: int = train_cfg["walk_forward_min_train"]
    test_ratio: float = train_cfg["test_ratio"]
    window_size: int = model_cfg["window_size"]
    feature_cols = MODEL_FEATURE_COLS

    # Carrega o parquet e computa índices válidos uma única vez
    df = load_features_parquet()
    valid_indices = compute_valid_indices(df, window_size, feature_cols)

    test_indices, folds = compute_walk_forward_splits(
        valid_indices, n_splits, test_ratio, min_train
    )

    # Constrói o test set uma única vez (independente dos folds)
    X_test_raw, y_test_raw = build_windows(df, test_indices, window_size, feature_cols)
    y_test_remapped = remap_labels(y_test_raw)

    best_fold_f1: float = -1.0
    best_fold_ckpt: Path | None = None
    best_fold_scaler = None

    logger.info(f"=== Walk-Forward CV: {len(folds)} folds | test={len(test_indices)} ===")

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        logger.info(
            f"\n--- Fold {fold_idx + 1}/{len(folds)} | "
            f"train={len(train_idx)} | val={len(val_idx)} ---"
        )

        # Constrói arrays para esta fold
        X_train_raw, y_train_raw = build_windows(df, train_idx, window_size, feature_cols)
        X_val_raw, y_val_raw = build_windows(df, val_idx, window_size, feature_cols)

        # Scaler fitado apenas no treino desta fold (anti-leakage)
        fold_scaler = fit_scaler(X_train_raw)
        X_train = fold_scaler.transform(X_train_raw)
        X_val = fold_scaler.transform(X_val_raw)

        y_train = remap_labels(y_train_raw)
        y_val = remap_labels(y_val_raw)

        train_ds = AgRo3Dataset(X_train, y_train)
        val_ds = AgRo3Dataset(X_val, y_val)

        fold_class_weights = compute_class_weights(
            y_train=train_ds.y.numpy(),
            num_classes=model_cfg["num_classes"],
            device=device,
        )

        fold_model = build_mlp_from_config(cfg, input_dim=train_ds.input_dim)
        fold_device = _verify_mps(fold_model, train_ds.input_dim, device)
        fold_model = fold_model.to(fold_device)

        fold_ckpt = checkpoint_path.parent / f"mlp_fold{fold_idx}.pt"
        fold_trainer = Trainer(
            model=fold_model, config=cfg, device=fold_device,
            class_weights=fold_class_weights, scaler=fold_scaler,
        )

        fold_train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        fold_val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_metrics = fold_trainer.fit(fold_train_loader, fold_val_loader, fold_ckpt)
        logger.info(
            f"Fold {fold_idx + 1} concluída: "
            f"val_f1_macro={best_metrics.val_f1_macro:.4f} (época {best_metrics.epoch})"
        )

        if best_metrics.val_f1_macro > best_fold_f1:
            best_fold_f1 = best_metrics.val_f1_macro
            best_fold_ckpt = fold_ckpt
            best_fold_scaler = fold_scaler

    assert best_fold_ckpt is not None and best_fold_scaler is not None

    logger.info(
        f"\n=== Walk-Forward concluído | Melhor fold por val_f1: {best_fold_ckpt.name} "
        f"| val_f1_macro={best_fold_f1:.4f} ==="
    )

    # -----------------------------------------------------------------------
    # FINAL REFIT — retreina sobre TODO o pré-test com o scaler mais amplo.
    #
    # Problema da seleção pela "melhor fold":
    #   A fold com maior val_f1 tende a ser a que valida num regime parecido
    #   com o treino — não a que mais generaliza para o futuro. No nosso caso,
    #   a Fold 1 (treino 2006-2009, val 2009-2012) ganhou, mas seu scaler é
    #   fitado em apenas ~200 amostras do período de CDI baixo histórico,
    #   incompatível com o test set de 2021-2025 (CDI 13-14%).
    #
    # Solução: após o CV (que serve para diagnóstico de regimes), retreinar um
    # modelo do zero sobre os primeiros 80% do pré-test e validar sobre os
    # últimos 20% (período mais recente, temporal proxy do test). O scaler
    # cobre ~568 amostras (2006-2019), normalizando CDI e preços em range mais
    # representativo para inferência futura.
    # -----------------------------------------------------------------------
    import shutil

    n_pre_test = len(valid_indices) - len(test_indices)
    pre_test_indices = valid_indices[:n_pre_test]

    n_rf_val = max(50, int(n_pre_test * 0.20))   # ≥ 50 amostras, ≈ 20%
    rf_train_idx = pre_test_indices[:-n_rf_val]
    rf_val_idx   = pre_test_indices[-n_rf_val:]

    logger.info(
        f"\n=== FINAL REFIT | train={len(rf_train_idx)} | val={len(rf_val_idx)} "
        f"(scaler fitado em {len(rf_train_idx)} amostras — cobre 2006→~2019) ==="
    )

    X_rf_train_raw, y_rf_train_raw = build_windows(df, rf_train_idx, window_size, feature_cols)
    X_rf_val_raw,   y_rf_val_raw   = build_windows(df, rf_val_idx,   window_size, feature_cols)

    rf_scaler    = fit_scaler(X_rf_train_raw)
    X_rf_train   = rf_scaler.transform(X_rf_train_raw)
    X_rf_val     = rf_scaler.transform(X_rf_val_raw)

    y_rf_train = remap_labels(y_rf_train_raw)
    y_rf_val   = remap_labels(y_rf_val_raw)

    rf_train_ds = AgRo3Dataset(X_rf_train, y_rf_train)
    rf_val_ds   = AgRo3Dataset(X_rf_val,   y_rf_val)

    rf_class_weights = compute_class_weights(
        y_train=rf_train_ds.y.numpy(),
        num_classes=model_cfg["num_classes"],
        device=device,
    )

    rf_model  = build_mlp_from_config(cfg, input_dim=rf_train_ds.input_dim)
    rf_device = _verify_mps(rf_model, rf_train_ds.input_dim, device)
    rf_model  = rf_model.to(rf_device)

    rf_ckpt = checkpoint_path.parent / "mlp_final_refit.pt"
    rf_trainer = Trainer(
        model=rf_model, config=cfg, device=rf_device,
        class_weights=rf_class_weights, scaler=rf_scaler,
    )

    rf_train_loader = DataLoader(rf_train_ds, batch_size=batch_size, shuffle=True)
    rf_val_loader   = DataLoader(rf_val_ds,   batch_size=batch_size, shuffle=False)

    rf_best = rf_trainer.fit(rf_train_loader, rf_val_loader, rf_ckpt)
    logger.info(
        f"Final Refit concluído: val_f1_macro={rf_best.val_f1_macro:.4f} "
        f"(época {rf_best.epoch})"
    )

    # O artefato de produção é o checkpoint do refit, não da melhor fold de CV
    shutil.copy2(rf_ckpt, checkpoint_path)
    logger.info(f"Final Refit checkpoint copiado para: {checkpoint_path}")

    # Avalia no test com o scaler do refit (normalizações do período mais longo)
    X_test   = rf_scaler.transform(X_test_raw)
    test_ds  = AgRo3Dataset(X_test, y_test_remapped)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    ckpt_rf    = Trainer.load_checkpoint(checkpoint_path, device=device)
    eval_model = build_mlp_from_config(cfg, input_dim=test_ds.input_dim)
    eval_model.load_state_dict(ckpt_rf["model_state_dict"])
    eval_model = eval_model.to(device)

    dummy_weights = torch.ones(model_cfg["num_classes"], device=device)
    eval_trainer  = Trainer(
        model=eval_model, config=cfg, device=device,
        class_weights=dummy_weights, scaler=rf_scaler,
    )
    _evaluate_on_test(
        eval_trainer, eval_model, checkpoint_path,
        test_loader, ckpt_rf["best_epoch"], device,
    )
    logger.info(f"Artefato final salvo em: {checkpoint_path.resolve()}")


def main() -> None:
    """Entry point: uv run python -m src.models.trainer

    Detecta automaticamente o modo de treinamento via model_config.yaml:
    - walk_forward_n_splits > 0 → Walk-Forward CV (recomendado)
    - walk_forward_n_splits = 0 → Split cronológico único (modo legado)
    """
    cfg = load_model_config()

    seed: int = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"Seeds definidos: {seed}")

    device = _select_device()
    logger.info(f"Device selecionado: {device}")

    checkpoint_path = Path("data/models/mlp_v1.pt")
    n_wf_splits: int = cfg["training"].get("walk_forward_n_splits", 0)

    if n_wf_splits > 0:
        logger.info(f"Modo: Walk-Forward CV ({n_wf_splits} folds)")
        _run_walk_forward(cfg, device, checkpoint_path)
    else:
        logger.info("Modo: Split cronológico único (legado)")
        _run_single_split(cfg, device, checkpoint_path)


if __name__ == "__main__":
    main()
