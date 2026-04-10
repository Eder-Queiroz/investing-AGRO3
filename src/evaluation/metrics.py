"""
Statistical Model Evaluation — Fase 5: Validação Estatística do Modelo AGRO3.

## Responsabilidade

Computa métricas de avaliação estatística para o classificador MLP treinado:
- F1-Score Macro, Precision (BUY), Recall (SELL) — métricas requeridas por CLAUDE.md
- MCC, Cohen's Kappa, Accuracy — métricas de suporte
- Bootstrap CIs com método BCa (corrige assimetria do F1-Macro em N≈167)
- Teste de permutação para significância estatística (H₀: modelo = chance)
- Baseline de maioria (always-predict-HOLD) como âncora de comparação

## Decisões de Design

- **`compute_metrics()` é função pura**: arrays in → `EvaluationReport` out.
  Sem modelo, sem checkpoint, sem DataLoader. Phase 7's `backtester.py`
  pode chamar a mesma função com seus próprios arrays de predição.

- **BCa sobre Percentile**: F1-Macro tem distribuição assimétrica à direita
  (limitada por 1.0) em N≈167. BCa corrige esse viés; percentile
  sistematicamente subestimaria o intervalo superior.

- **Permutação embaralha y_pred, não y_true**: testa "os assignments
  específicos do modelo são informativos?" preservando a distribuição
  marginal das predições.

- **`labels=[0,1,2]` na confusion_matrix**: fixa shape (3,3) mesmo quando
  uma classe está ausente das predições — sklearn retorna (2,2) sem isso.

- **`per_class` keyed by int**: `report.per_class[2].precision` evita
  risco de off-by-one com indexação por lista.

## Uso

    from src.evaluation.metrics import compute_metrics, evaluate_from_checkpoint

    # Avaliação pura (sem checkpoint)
    report = compute_metrics(y_true, y_pred, split_name="test")

    # Avaliação a partir do checkpoint treinado
    reports = evaluate_from_checkpoint(Path("data/models/mlp_v1.pt"))
    print_report(reports["test"])

    # CLI
    uv run python -m src.evaluation.metrics
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

CLASS_NAMES: dict[int, str] = {0: "SELL", 1: "HOLD", 2: "BUY"}


@dataclasses.dataclass(frozen=True)
class ClassMetrics:
    """Métricas de precisão/recall/F1 para uma única classe.

    Atributos:
        class_idx: Índice da classe (0=SELL, 1=HOLD, 2=BUY).
        class_name: Nome legível da classe.
        precision: Fração de predições desta classe que estão corretas.
        recall: Fração de instâncias reais desta classe que foram detectadas.
        f1: Média harmônica de precision e recall.
        support: Número de instâncias reais desta classe no conjunto avaliado.
    """

    class_idx: int
    class_name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclasses.dataclass(frozen=True)
class BootstrapCI:
    """Intervalo de confiança bootstrap com método BCa.

    Atributos:
        lower: Limite inferior do IC (método BCa).
        upper: Limite superior do IC (método BCa).
        confidence_level: Nível de confiança (ex: 0.95).
        point_estimate: Estimativa pontual no conjunto completo (NOT média bootstrap).
            BCa não centraliza o IC na estimativa pontual — armazenar ambos é
            necessário para o printer exibir valores consistentes.
    """

    lower: float
    upper: float
    confidence_level: float
    point_estimate: float


@dataclasses.dataclass(frozen=True)
class CheckpointMeta:
    """Metadados extraídos do checkpoint de treinamento.

    Atributos:
        best_epoch: Época em que o melhor F1-Macro de validação foi atingido.
        best_val_f1: Melhor F1-Macro de validação registrado.
        total_epochs_trained: Total de épocas executadas (incluindo após best_epoch).
        class_weights: Pesos de classe usados no CrossEntropyLoss (ou None).
    """

    best_epoch: int
    best_val_f1: float
    total_epochs_trained: int
    class_weights: list[float] | None


@dataclasses.dataclass
class EvaluationReport:
    """Relatório completo de avaliação estatística para um split.

    Atributos:
        split_name: Nome do split avaliado (ex: "val", "test").
        n_samples: Número total de amostras avaliadas.

        f1_macro: F1-Score macro-averaged — métrica primária do CLAUDE.md.
        precision_buy: Precision da classe BUY (2) — custo de falso positivo.
        recall_sell: Recall da classe SELL (0) — custo de falso negativo.

        accuracy: Acurácia global (enganosa com classes desbalanceadas).
        mcc: Matthews Correlation Coefficient — robusto a desbalanceamento.
        cohen_kappa: Concordância ajustada ao acaso.
        majority_baseline_f1: F1-Macro do classificador que sempre prediz HOLD.

        per_class: Métricas por classe; indexado por int (0, 1, 2).
        confusion_matrix: Matriz (3×3) — linhas=real, colunas=predito.

        ci_f1_macro: IC 95% BCa para F1-Macro.
        ci_precision_buy: IC 95% BCa para Precision(BUY).
        ci_recall_sell: IC 95% BCa para Recall(SELL).

        p_value_vs_baseline: p-value do teste de permutação (H₀: modelo = chance).
        is_significant: True se p_value_vs_baseline < 0.05 (strict).
        checkpoint_meta: Metadados do checkpoint (None se avaliação sem checkpoint).
    """

    split_name: str
    n_samples: int
    # --- CLAUDE.md required ---
    f1_macro: float
    precision_buy: float
    recall_sell: float
    # --- Supporting ---
    accuracy: float
    mcc: float
    cohen_kappa: float
    majority_baseline_f1: float
    # --- Breakdown ---
    per_class: dict[int, ClassMetrics]
    confusion_matrix: np.ndarray  # shape (3, 3)
    # --- Bootstrap CIs (95%) ---
    ci_f1_macro: BootstrapCI
    ci_precision_buy: BootstrapCI
    ci_recall_sell: BootstrapCI
    # --- Significance ---
    p_value_vs_baseline: float
    is_significant: bool
    # --- Training context ---
    checkpoint_meta: CheckpointMeta | None


# ---------------------------------------------------------------------------
# Pure Metric Functions (no PyTorch, no checkpoint)
# ---------------------------------------------------------------------------


def compute_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_idx: int,
    class_name: str,
) -> ClassMetrics:
    """Computa precision, recall, F1 e support para uma classe específica.

    Usa `labels=[class_idx]` com `average=None` para calcular métricas
    binárias corretas para qualquer classe (funciona para BUY=2, não apenas
    para a classe padrão positive=1 de `average='binary'`).

    Args:
        y_true: Array de labels reais em {0, 1, 2}.
        y_pred: Array de labels preditos em {0, 1, 2}.
        class_idx: Índice da classe a avaliar (0, 1, ou 2).
        class_name: Nome legível da classe (ex: "BUY").

    Returns:
        ClassMetrics com todas as métricas da classe especificada.
    """
    precision = float(
        precision_score(
            y_true, y_pred, labels=[class_idx], average=None, zero_division=0
        )[0]
    )
    recall = float(
        recall_score(
            y_true, y_pred, labels=[class_idx], average=None, zero_division=0
        )[0]
    )
    f1 = float(
        f1_score(y_true, y_pred, labels=[class_idx], average=None, zero_division=0)[0]
    )
    support = int(np.sum(y_true == class_idx))

    return ClassMetrics(
        class_idx=class_idx,
        class_name=class_name,
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
    )


def compute_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Computa IC bootstrap com método BCa para uma métrica arbitrária.

    Usa `scipy.stats.bootstrap` com `method='BCa'` e `paired=True` para
    preservar o alinhamento (y_true[i], y_pred[i]) durante o reamostramento.

    BCa (bias-corrected and accelerated) é preferido ao percentile para
    métricas como F1-Macro que têm distribuição assimétrica em N≈167.

    Args:
        y_true: Array de labels reais.
        y_pred: Array de labels preditos.
        metric_fn: Função (y_true, y_pred) → float a ser bootstrappada.
        n_bootstrap: Número de amostras bootstrap (padrão: 1000).
        confidence: Nível de confiança do IC (padrão: 0.95).

    Returns:
        BootstrapCI com lower, upper, confidence_level e point_estimate.
        Em caso de colapso de classe (métrica constante), retorna IC degenerado
        lower=upper=point_estimate com aviso de log.
    """
    point_estimate = metric_fn(y_true, y_pred)

    # scipy.stats.bootstrap requer que statistic retorne array, não scalar
    def _stat(y_t: np.ndarray, y_p: np.ndarray) -> np.ndarray:
        return np.array([metric_fn(y_t, y_p)])

    try:
        result = scipy.stats.bootstrap(
            data=(y_true, y_pred),
            statistic=_stat,
            n_resamples=n_bootstrap,
            method="BCa",
            paired=True,
            confidence_level=confidence,
            random_state=42,
        )
        lower = float(result.confidence_interval.low[0])
        upper = float(result.confidence_interval.high[0])

        # BCa pode produzir NaN quando a métrica é constante em todas as amostras
        if np.isnan(lower) or np.isnan(upper):
            raise ValueError("BCa produziu NaN — métrica provavelmente constante")

    except (ValueError, Exception) as e:
        logger.warning(
            f"BCa bootstrap falhou ({e}). "
            f"Usando IC degenerado lower=upper=point_estimate={point_estimate:.4f}. "
            f"Provável causa: modelo colapsou para uma única classe."
        )
        lower = point_estimate
        upper = point_estimate

    # Garante que lower ≤ upper (BCa pode inverter em casos extremos)
    if lower > upper:
        lower, upper = upper, lower

    return BootstrapCI(
        lower=lower,
        upper=upper,
        confidence_level=confidence,
        point_estimate=point_estimate,
    )


def compute_permutation_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_permutations: int = 1000,
    seed: int = 42,
) -> float:
    """Testa a significância estatística do modelo vs. baseline aleatório.

    H₀: As predições do modelo não são mais informativas do que atribuições
    aleatórias da mesma distribuição de classes.

    Embaralha `y_pred` (não `y_true`) para preservar a distribuição marginal
    das predições do modelo enquanto quebra o alinhamento estrutural com os
    labels reais.

    Usa `np.random.default_rng(seed)` (Generator API) para não poluir o
    estado global do gerador.

    Args:
        y_true: Array de labels reais em {0, 1, 2}.
        y_pred: Array de labels preditos em {0, 1, 2}.
        n_permutations: Número de permutações (padrão: 1000).
        seed: Semente para reprodutibilidade (padrão: 42).

    Returns:
        p-value: Fração de permutações com F1-Macro ≥ F1-Macro observado.
        Interpretar como: p < 0.05 → rejeita H₀ → modelo é significativo.
    """
    observed = f1_score(y_true, y_pred, average="macro", zero_division=0)
    rng = np.random.default_rng(seed)

    null_distribution = [
        f1_score(y_true, rng.permutation(y_pred), average="macro", zero_division=0)
        for _ in range(n_permutations)
    ]

    p_value = float(np.mean(np.array(null_distribution) >= observed))
    return p_value


def compute_majority_baseline_f1(y_true: np.ndarray) -> float:
    """Computa o F1-Macro de um classificador que sempre prediz a classe majoritária.

    Em dados dominados por HOLD (~60-70%), este baseline produz F1-Macro baixo
    (~0.1-0.3) porque F1=0 para SELL e BUY arrastam a média para baixo.
    Isso demonstra por que acurácia (~60-70%) é enganosa.

    Args:
        y_true: Array de labels reais em {0, 1, 2}.

    Returns:
        F1-Macro do classificador de maioria como float.
    """
    majority_class = int(np.bincount(y_true).argmax())
    y_baseline = np.full_like(y_true, majority_class)
    return float(f1_score(y_true, y_baseline, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Main Assembly Function (pure)
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
    split_name: str = "unknown",
    checkpoint_meta: CheckpointMeta | None = None,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
    seed: int = 42,
) -> EvaluationReport:
    """Computa o relatório completo de avaliação estatística.

    Função pura: arrays in → EvaluationReport out. Sem modelo, sem checkpoint,
    sem DataLoader. O backtester.py da Fase 7 pode chamar esta mesma função
    com seus próprios arrays de predição.

    Args:
        y_true: Array de labels reais em {0, 1, 2}, shape (N,).
        y_pred: Array de labels preditos em {0, 1, 2}, shape (N,).
        y_proba: Probabilidades por classe, shape (N, 3). Aceito para
            estabilidade de API da Fase 7 — ignorado nesta fase com log.debug.
        split_name: Nome do split avaliado (ex: "val", "test").
        checkpoint_meta: Metadados do checkpoint de treinamento (opcional).
        n_bootstrap: Número de amostras bootstrap para os ICs (padrão: 1000).
        n_permutations: Número de permutações para o teste de significância.
        seed: Semente para reprodutibilidade do bootstrap e permutação.

    Returns:
        EvaluationReport com todas as métricas, ICs e metadados.

    Raises:
        ValueError: Se y_true e y_pred têm shapes incompatíveis ou contêm
            valores fora de {0, 1, 2}.
    """
    # --- Validação de entrada ---
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true e y_pred devem ter o mesmo shape. "
            f"y_true: {y_true.shape}, y_pred: {y_pred.shape}"
        )
    if y_true.ndim != 1:
        raise ValueError(f"y_true deve ser 1D. Shape recebido: {y_true.shape}")

    valid_labels = {0, 1, 2}
    invalid_true = set(np.unique(y_true)) - valid_labels
    invalid_pred = set(np.unique(y_pred)) - valid_labels
    if invalid_true:
        raise ValueError(f"y_true contém valores fora de {{0,1,2}}: {invalid_true}")
    if invalid_pred:
        raise ValueError(f"y_pred contém valores fora de {{0,1,2}}: {invalid_pred}")

    if y_proba is not None:
        y_proba_arr = np.asarray(y_proba)
        if y_proba_arr.shape != (len(y_true), 3):
            raise ValueError(
                f"y_proba deve ter shape (N, 3). Recebido: {y_proba_arr.shape}"
            )
        logger.debug(
            "y_proba recebido mas ignorado nesta fase — calibração reservada para Fase 7."
        )

    n_samples = len(y_true)
    logger.info(f"Computando métricas para split='{split_name}' com N={n_samples}")

    # --- 1. Métricas escalares ---
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    accuracy = float(accuracy_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    cohen_kappa = float(cohen_kappa_score(y_true, y_pred))
    majority_baseline_f1 = compute_majority_baseline_f1(y_true)

    # Métricas requeridas por CLAUDE.md
    precision_buy = float(
        precision_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]
    )
    recall_sell = float(
        recall_score(y_true, y_pred, labels=[0], average=None, zero_division=0)[0]
    )

    # --- 2. Confusion matrix — labels explícito para fixar shape (3×3) ---
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    # --- 3. Métricas por classe ---
    per_class: dict[int, ClassMetrics] = {
        idx: compute_class_metrics(y_true, y_pred, idx, name)
        for idx, name in CLASS_NAMES.items()
    }

    # --- 4. Bootstrap CIs (BCa) ---
    def _f1_macro_fn(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(f1_score(yt, yp, average="macro", zero_division=0))

    def _precision_buy_fn(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(
            precision_score(yt, yp, labels=[2], average=None, zero_division=0)[0]
        )

    def _recall_sell_fn(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(
            recall_score(yt, yp, labels=[0], average=None, zero_division=0)[0]
        )

    logger.info("Computando bootstrap CIs (BCa, N_bootstrap=%d)...", n_bootstrap)
    ci_f1_macro = compute_bootstrap_ci(y_true, y_pred, _f1_macro_fn, n_bootstrap)
    ci_precision_buy = compute_bootstrap_ci(
        y_true, y_pred, _precision_buy_fn, n_bootstrap
    )
    ci_recall_sell = compute_bootstrap_ci(y_true, y_pred, _recall_sell_fn, n_bootstrap)

    # --- 5. Permutation test ---
    logger.info(
        "Executando teste de permutação (N_permutations=%d)...", n_permutations
    )
    p_value = compute_permutation_test(y_true, y_pred, n_permutations, seed)
    is_significant = p_value < 0.05  # strict <, convenção estatística

    return EvaluationReport(
        split_name=split_name,
        n_samples=n_samples,
        f1_macro=f1_macro,
        precision_buy=precision_buy,
        recall_sell=recall_sell,
        accuracy=accuracy,
        mcc=mcc,
        cohen_kappa=cohen_kappa,
        majority_baseline_f1=majority_baseline_f1,
        per_class=per_class,
        confusion_matrix=cm,
        ci_f1_macro=ci_f1_macro,
        ci_precision_buy=ci_precision_buy,
        ci_recall_sell=ci_recall_sell,
        p_value_vs_baseline=p_value,
        is_significant=is_significant,
        checkpoint_meta=checkpoint_meta,
    )


# ---------------------------------------------------------------------------
# Checkpoint-Aware Orchestrator
# ---------------------------------------------------------------------------


def evaluate_from_checkpoint(
    checkpoint_path: Path,
    splits: list[str] | None = None,
    device: torch.device | None = None,
    n_bootstrap: int = 1000,
    n_permutations: int = 1000,
) -> dict[str, EvaluationReport]:
    """Avalia o modelo salvo no checkpoint em um ou mais splits.

    Orquestra: carregamento do checkpoint → reconstrução do modelo →
    carregamento dos dados → inferência em batches → compute_metrics.

    Args:
        checkpoint_path: Caminho para o arquivo .pt do checkpoint. Deve conter
            as chaves: model_state_dict, config, input_dim, metrics_history,
            best_epoch, best_val_f1, class_weights.
        splits: Lista de splits a avaliar (ex: ["val", "test"]).
            Padrão: ["val", "test"].
        device: Dispositivo PyTorch. Se None, auto-detecta: cuda → mps → cpu.
        n_bootstrap: Número de amostras bootstrap para ICs (padrão: 1000).
        n_permutations: Número de permutações para o teste de significância.

    Returns:
        Dicionário split_name → EvaluationReport.

    Raises:
        FileNotFoundError: Se checkpoint_path não existir. Também propagará
            FileNotFoundError se `features_weekly.parquet` não existir
            (error de create_datasets, não do checkpoint).
        KeyError: Se o checkpoint não contiver as chaves esperadas.
    """
    # Importações lazy para evitar ciclo de importação (evaluation ← models ← ...)
    from src.models.dataset import create_datasets
    from src.models.mlp import build_mlp_from_config
    from src.models.trainer import Trainer

    if splits is None:
        splits = ["val", "test"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {checkpoint_path}. "
            f"Execute 'uv run python -m src.models.trainer' primeiro."
        )

    # --- Device auto-detection (mesma lógica do trainer.py) ---
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            try:
                if torch.backends.mps.is_available():
                    _probe = torch.zeros(1, device="mps") + torch.zeros(1, device="mps")
                    device = torch.device("mps")
                else:
                    device = torch.device("cpu")
            except RuntimeError:
                device = torch.device("cpu")

    logger.info(f"Dispositivo selecionado: {device}")

    # --- Carrega checkpoint ---
    ckpt = Trainer.load_checkpoint(checkpoint_path, device)
    config: dict[str, Any] = ckpt["config"]
    input_dim: int = ckpt["input_dim"]

    # --- Reconstrói modelo ---
    model = build_mlp_from_config(config, input_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        f"Modelo reconstruído: input_dim={input_dim} | "
        f"best_epoch={ckpt['best_epoch']} | best_val_f1={ckpt['best_val_f1']:.4f}"
    )

    # --- Carrega dados ---
    train_ds, val_ds, test_ds, _scaler = create_datasets(config=config)

    split_datasets = {"train": train_ds, "val": val_ds, "test": test_ds}

    # --- CheckpointMeta ---
    class_weights_raw = ckpt.get("class_weights")
    class_weights_list: list[float] | None = None
    if class_weights_raw is not None:
        if isinstance(class_weights_raw, torch.Tensor):
            class_weights_list = class_weights_raw.cpu().tolist()
        else:
            class_weights_list = list(class_weights_raw)

    meta = CheckpointMeta(
        best_epoch=int(ckpt["best_epoch"]),
        best_val_f1=float(ckpt["best_val_f1"]),
        total_epochs_trained=len(ckpt["metrics_history"]),
        class_weights=class_weights_list,
    )

    # --- Inferência e avaliação por split ---
    reports: dict[str, EvaluationReport] = {}

    for split_name in splits:
        if split_name not in split_datasets:
            logger.warning(f"Split '{split_name}' desconhecido — ignorado.")
            continue

        ds = split_datasets[split_name]
        if len(ds) == 0:
            logger.warning(f"Split '{split_name}' está vazio — ignorado.")
            continue

        from torch.utils.data import DataLoader

        loader = DataLoader(ds, batch_size=256, shuffle=False)

        all_preds: list[np.ndarray] = []
        all_targets: list[np.ndarray] = []
        all_proba: list[np.ndarray] = []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                logits = model(X_batch)
                proba = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_preds.append(preds.cpu().numpy())
                all_targets.append(y_batch.numpy())
                all_proba.append(proba.cpu().numpy())

        y_pred_arr = np.concatenate(all_preds)
        y_true_arr = np.concatenate(all_targets)
        y_proba_arr = np.concatenate(all_proba)

        logger.info(
            f"Split '{split_name}': {len(y_true_arr)} amostras — "
            f"iniciando compute_metrics..."
        )

        report = compute_metrics(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            y_proba=y_proba_arr,
            split_name=split_name,
            checkpoint_meta=meta,
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
        )
        reports[split_name] = report

    return reports


# ---------------------------------------------------------------------------
# Console Printer
# ---------------------------------------------------------------------------


def print_report(report: EvaluationReport) -> None:
    """Imprime o relatório de avaliação no console em formato ASCII.

    Usa print() (não logger) pois é um relatório de console, não um evento
    de log. Não requer bibliotecas externas de visualização.

    Args:
        report: EvaluationReport a ser exibido.
    """
    sep = "=" * 62
    thin = "-" * 40

    split_upper = report.split_name.upper()
    print(sep)
    print(f"  EVALUATION REPORT: {split_upper}  (N={report.n_samples} samples)")
    print(sep)

    # --- CLAUDE.md required metrics ---
    print()
    print("  CLAUDE.md REQUIRED METRICS")
    print(f"  {thin[:28]}")

    ci = report.ci_f1_macro
    print(
        f"  F1-Score Macro    : {report.f1_macro:>7.4f}  "
        f"[95% CI: {ci.lower:.4f} – {ci.upper:.4f}]"
    )
    ci = report.ci_precision_buy
    print(
        f"  Precision (BUY)   : {report.precision_buy:>7.4f}  "
        f"[95% CI: {ci.lower:.4f} – {ci.upper:.4f}]"
    )
    ci = report.ci_recall_sell
    print(
        f"  Recall    (SELL)  : {report.recall_sell:>7.4f}  "
        f"[95% CI: {ci.lower:.4f} – {ci.upper:.4f}]"
    )

    # --- Supporting metrics ---
    print()
    print("  SUPPORTING METRICS")
    print(f"  {thin[:18]}")
    print(f"  Accuracy          : {report.accuracy:>7.4f}")
    print(f"  Matthews CC (MCC) : {report.mcc:>7.4f}")
    print(f"  Cohen's Kappa     : {report.cohen_kappa:>7.4f}")
    print(
        f"  Baseline F1-Macro : {report.majority_baseline_f1:>7.4f}"
        f"  (always-predict-HOLD)"
    )

    # --- Significance test ---
    print()
    print("  SIGNIFICANCE TEST")
    print(f"  {thin[:17]}")
    print(f"  Permutation p-val : {report.p_value_vs_baseline:>7.4f}")
    sig_str = "YES (p < 0.05)" if report.is_significant else "NO  (p >= 0.05)"
    print(f"  Significant?      :  {sig_str}")

    # --- Per-class breakdown ---
    print()
    print("  PER-CLASS BREAKDOWN")
    print(f"  {thin[:19]}")
    print(f"  {'Class':<10} {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'Support':>7}")
    print(f"  {'-'*10} {'-'*6}  {'-'*6}  {'-'*6}  {'-'*7}")
    for idx in [0, 1, 2]:
        cm_cls = report.per_class[idx]
        print(
            f"  {cm_cls.class_name+f' ({idx})':<10} "
            f"{cm_cls.precision:>6.3f}  "
            f"{cm_cls.recall:>6.3f}  "
            f"{cm_cls.f1:>6.3f}  "
            f"{cm_cls.support:>7d}"
        )

    # --- Confusion matrix ---
    print()
    print("  CONFUSION MATRIX  (rows=true, cols=pred)")
    print(f"  {thin[:40]}")
    print(f"  {'':8}  {'SELL':>6}  {'HOLD':>6}  {'BUY':>6}")
    class_labels = ["SELL", "HOLD", "BUY"]
    for i, label in enumerate(class_labels):
        row = report.confusion_matrix[i]
        print(f"  {label:<8}  {row[0]:>6d}  {row[1]:>6d}  {row[2]:>6d}")

    # --- Training context (omitido se checkpoint_meta is None) ---
    if report.checkpoint_meta is not None:
        meta = report.checkpoint_meta
        print()
        print("  TRAINING CONTEXT")
        print(f"  {thin[:16]}")
        print(f"  Best epoch         : {meta.best_epoch}")
        print(f"  Best val F1-Macro  : {meta.best_val_f1:.4f}")
        print(f"  Total epochs       : {meta.total_epochs_trained}")

    print(sep)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


def main() -> None:
    """Avalia o checkpoint treinado em val e test e imprime os relatórios.

    Uso:
        uv run python -m src.evaluation.metrics

    Requer:
        data/models/mlp_v1.pt — gerado por 'uv run python -m src.models.trainer'
    """
    checkpoint_path = Path("data/models/mlp_v1.pt")

    logger.info(f"Iniciando avaliação a partir de: {checkpoint_path}")

    reports = evaluate_from_checkpoint(
        checkpoint_path=checkpoint_path,
        splits=["val", "test"],
    )

    for split_name in ["val", "test"]:
        if split_name in reports:
            print()
            print_report(reports[split_name])
        else:
            logger.warning(f"Nenhum relatório gerado para split='{split_name}'.")


if __name__ == "__main__":
    main()
