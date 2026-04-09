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

from src.models.dataset import create_datasets
from src.models.mlp import build_mlp_from_config
from src.utils.config import load_model_config
from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# Mapeamento de índice → rótulo legível
_CLASS_NAMES: dict[int, str] = {0: "SELL", 1: "HOLD", 2: "BUY"}


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
    val_f1_hold: float
    val_f1_buy: float

    def log_summary(self) -> None:
        """Emite uma linha de log INFO com o resumo da época."""
        logger.info(
            f"Epoch {self.epoch:03d} | "
            f"train_loss={self.train_loss:.4f} | "
            f"val_loss={self.val_loss:.4f} | "
            f"val_acc={self.val_acc:.3f} | "
            f"f1_macro={self.val_f1_macro:.4f} | "
            f"f1[SELL={self.val_f1_sell:.3f} "
            f"HOLD={self.val_f1_hold:.3f} "
            f"BUY={self.val_f1_buy:.3f}]"
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
        num_classes: Total de classes (3 para SELL/HOLD/BUY).
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
            val_f1_hold=float(f1_per_class[1]),
            val_f1_buy=float(f1_per_class[2]),
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
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        logger.info(f"Confusion Matrix [{label}]:")
        logger.info("  Classes: SELL(0), HOLD(1), BUY(2)")
        logger.info("  Pred →    SELL   HOLD    BUY")
        for i, row in enumerate(cm):
            logger.info(f"  {_CLASS_NAMES[i]:6s} ↓  {row[0]:5d}  {row[1]:5d}  {row[2]:5d}")


def _get_input_dim(model: nn.Module) -> int:
    """Extrai input_dim do primeiro Linear do modelo."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            return module.in_features
    return -1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: uv run python -m src.models.trainer

    Executa o pipeline completo:
      1. Carrega config e define seeds
      2. Detecta device (cuda → mps → cpu)
      3. Cria datasets via create_datasets()
      4. Computa class weights do train
      5. Instancia modelo e Trainer
      6. Treina com early stopping
      7. Avalia no test set com o melhor checkpoint
    """
    cfg = load_model_config()

    # Reprodutibilidade
    seed: int = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"Seeds definidos: {seed}")

    # Seleção de device: cuda → mps (Apple Silicon) → cpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Device selecionado: {device}")

    # Datasets
    train_ds, val_ds, test_ds, scaler = create_datasets(config=cfg)

    # Class weights computados exclusivamente do train (anti-leakage)
    class_weights = compute_class_weights(
        y_train=train_ds.y.numpy(),
        num_classes=cfg["model"]["num_classes"],
        device=device,
    )

    batch_size: int = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Modelo
    model = build_mlp_from_config(cfg, input_dim=train_ds.input_dim)

    # Verificação de compatibilidade MPS (some ops podem não ser suportadas)
    if device.type == "mps":
        try:
            _probe = torch.randn(2, train_ds.input_dim, device=device)
            model.to(device)
            _ = model(_probe)
            logger.info("MPS: forward pass de verificação OK.")
        except RuntimeError as exc:
            logger.warning(
                f"MPS indisponível para este modelo ({exc}). Fallback para CPU."
            )
            device = torch.device("cpu")
            model = model.cpu()
    else:
        model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Modelo pronto: {n_params:,} parâmetros | "
        f"input_dim={train_ds.input_dim} | "
        f"train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}"
    )

    # Treinamento
    trainer = Trainer(
        model=model,
        config=cfg,
        device=device,
        class_weights=class_weights,
        scaler=scaler,
    )

    checkpoint_path = Path("data/models/mlp_v1.pt")
    best_metrics = trainer.fit(train_loader, val_loader, checkpoint_path)

    # Avaliação final no test set com o melhor checkpoint
    logger.info("=== Avaliação no Test Set (melhor checkpoint) ===")
    checkpoint = Trainer.load_checkpoint(checkpoint_path, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_metrics = trainer._val_epoch(
        test_loader,
        epoch=best_metrics.epoch,
        train_loss=0.0,
    )
    trainer._log_confusion_matrix(test_loader, label="test (final)")

    logger.info(
        f"TEST FINAL | "
        f"loss={test_metrics.val_loss:.4f} | "
        f"acc={test_metrics.val_acc:.3f} | "
        f"f1_macro={test_metrics.val_f1_macro:.4f} | "
        f"f1[SELL={test_metrics.val_f1_sell:.3f} "
        f"HOLD={test_metrics.val_f1_hold:.3f} "
        f"BUY={test_metrics.val_f1_buy:.3f}]"
    )
    logger.info(f"Artefato salvo em: {checkpoint_path.resolve()}")


if __name__ == "__main__":
    main()
