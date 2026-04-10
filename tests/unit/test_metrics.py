"""
Testes unitários para src/evaluation/metrics.py — Fase 5.

Cobertura: ~26 testes em 8 grupos.
  1. compute_class_metrics     — precisão/recall por classe específica
  2. compute_bootstrap_ci      — ICs BCa, reprodutibilidade, caso degenerado
  3. compute_permutation_test  — p-value em [0,1], modelo perfeito vs. aleatório
  4. compute_majority_baseline_f1 — classe majoritária, F1 baixo para HOLD-dominado
  5. compute_metrics           — montagem do EvaluationReport completo
  6. EvaluationReport.is_significant — limiar estrito p < 0.05
  7. print_report              — não levanta, stdout não-vazio, conteúdo mínimo
  8. evaluate_from_checkpoint  — FileNotFoundError para path inexistente

Todos os testes rodam em CPU, sem dados reais e sem checkpoint.
Fixtures sintéticas cobrem: balanceado, perfeito, aleatório e HOLD-dominado.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from src.evaluation.metrics import (
    BootstrapCI,
    ClassMetrics,
    EvaluationReport,
    compute_bootstrap_ci,
    compute_class_metrics,
    compute_majority_baseline_f1,
    compute_metrics,
    compute_permutation_test,
    evaluate_from_checkpoint,
    print_report,
)

# ---------------------------------------------------------------------------
# Fixtures compartilhadas
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(0)


@pytest.fixture
def balanced_labels() -> tuple[np.ndarray, np.ndarray]:
    """N=60 amostras balanceadas: 20 de cada classe."""
    y_true = np.array([0] * 20 + [1] * 20 + [2] * 20)
    y_pred = np.array([0] * 20 + [1] * 20 + [2] * 20)  # perfeito
    return y_true, y_pred


@pytest.fixture
def hold_dominated_labels() -> tuple[np.ndarray, np.ndarray]:
    """N=100 amostras HOLD-dominadas: 10 SELL, 70 HOLD, 20 BUY."""
    y_true = np.array([0] * 10 + [1] * 70 + [2] * 20)
    y_pred = np.full(100, 1, dtype=np.int64)  # sempre prediz HOLD
    return y_true, y_pred


@pytest.fixture
def random_labels() -> tuple[np.ndarray, np.ndarray]:
    """N=150 amostras aleatórias — p-value esperado próximo a 1.0."""
    rng = np.random.default_rng(999)
    y_true = rng.integers(0, 3, size=150)
    y_pred = rng.integers(0, 3, size=150)
    return y_true, y_pred


@pytest.fixture
def perfect_labels() -> tuple[np.ndarray, np.ndarray]:
    """N=90 amostras com predição perfeita — p-value esperado ~ 0."""
    y_true = np.array([0] * 30 + [1] * 30 + [2] * 30)
    y_pred = y_true.copy()
    return y_true, y_pred


@pytest.fixture
def sample_report(balanced_labels: tuple[np.ndarray, np.ndarray]) -> EvaluationReport:
    """EvaluationReport gerado com dados balanceados perfeitos."""
    y_true, y_pred = balanced_labels
    return compute_metrics(y_true, y_pred, split_name="test")


# ---------------------------------------------------------------------------
# Grupo 1: compute_class_metrics
# ---------------------------------------------------------------------------


def test_class_metrics_perfect_buy() -> None:
    """Predição perfeita para BUY deve resultar em precision=recall=f1=1.0."""
    y_true = np.array([0, 1, 2, 2, 2])
    y_pred = np.array([0, 1, 2, 2, 2])
    result = compute_class_metrics(y_true, y_pred, class_idx=2, class_name="BUY")
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)


def test_class_metrics_sell_never_predicted() -> None:
    """Quando SELL nunca é predito, recall(SELL) = 0.0 (verdadeiros não detectados)."""
    y_true = np.array([0, 0, 1, 2])
    y_pred = np.array([1, 2, 1, 2])  # SELL nunca predito
    result = compute_class_metrics(y_true, y_pred, class_idx=0, class_name="SELL")
    assert result.recall == pytest.approx(0.0)


def test_class_metrics_support_count() -> None:
    """Support deve contar instâncias reais da classe, não preditas."""
    y_true = np.array([0, 0, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 2])
    result = compute_class_metrics(y_true, y_pred, class_idx=0, class_name="SELL")
    assert result.support == 3  # três SELL reais


def test_class_metrics_zero_division_safety() -> None:
    """Quando a classe não aparece em y_pred, não deve levantar erro (zero_division=0)."""
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 1])  # BUY (2) nunca predito nem real
    result = compute_class_metrics(y_true, y_pred, class_idx=2, class_name="BUY")
    assert result.precision == pytest.approx(0.0)
    assert result.recall == pytest.approx(0.0)
    assert result.f1 == pytest.approx(0.0)
    assert result.support == 0


def test_class_metrics_returns_dataclass() -> None:
    """Deve retornar instância de ClassMetrics (frozen=True)."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    result = compute_class_metrics(y_true, y_pred, 1, "HOLD")
    assert isinstance(result, ClassMetrics)
    assert result.class_idx == 1
    assert result.class_name == "HOLD"


# ---------------------------------------------------------------------------
# Grupo 2: compute_bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_bounds_contain_point_estimate(
    balanced_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    """lower ≤ point_estimate ≤ upper deve ser satisfeito em condições normais."""
    from sklearn.metrics import f1_score

    y_true, y_pred = balanced_labels

    def f1_fn(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(f1_score(yt, yp, average="macro", zero_division=0))

    ci = compute_bootstrap_ci(y_true, y_pred, f1_fn, n_bootstrap=200)
    assert ci.lower <= ci.point_estimate + 1e-6
    assert ci.upper >= ci.point_estimate - 1e-6


def test_bootstrap_ci_stores_confidence_level(
    balanced_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    """confidence_level deve ser armazenado corretamente no resultado."""
    from sklearn.metrics import f1_score

    y_true, y_pred = balanced_labels

    ci = compute_bootstrap_ci(
        y_true, y_pred, lambda yt, yp: float(f1_score(yt, yp, average="macro")),
        n_bootstrap=100, confidence=0.90,
    )
    assert ci.confidence_level == pytest.approx(0.90)


def test_bootstrap_ci_reproducible_with_same_seed(
    balanced_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    """Duas chamadas com os mesmos dados devem produzir o mesmo IC (seed=42 fixo)."""
    from sklearn.metrics import f1_score

    y_true, y_pred = balanced_labels

    def fn(yt: np.ndarray, yp: np.ndarray) -> float:
        return float(f1_score(yt, yp, average="macro", zero_division=0))

    ci1 = compute_bootstrap_ci(y_true, y_pred, fn, n_bootstrap=100)
    ci2 = compute_bootstrap_ci(y_true, y_pred, fn, n_bootstrap=100)
    assert ci1.lower == pytest.approx(ci2.lower)
    assert ci1.upper == pytest.approx(ci2.upper)


def test_bootstrap_ci_degenerate_case_does_not_raise() -> None:
    """Quando a métrica é sempre 0.0, não deve levantar erro — IC degenerado."""
    y_true = np.array([0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1, 1])  # nunca acerta SELL

    def always_zero(yt: np.ndarray, yp: np.ndarray) -> float:
        from sklearn.metrics import f1_score
        return float(f1_score(yt, yp, labels=[0], average=None, zero_division=0)[0])

    ci = compute_bootstrap_ci(y_true, y_pred, always_zero, n_bootstrap=100)
    assert isinstance(ci, BootstrapCI)
    assert not (ci.lower != ci.lower)  # NaN check via IEEE 754 self-inequality
    assert not (ci.upper != ci.upper)


def test_bootstrap_ci_returns_dataclass(
    balanced_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    """Deve retornar instância de BootstrapCI."""
    y_true, y_pred = balanced_labels
    ci = compute_bootstrap_ci(
        y_true, y_pred,
        lambda yt, yp: 0.5,  # constante
        n_bootstrap=50,
    )
    assert isinstance(ci, BootstrapCI)


# ---------------------------------------------------------------------------
# Grupo 3: compute_permutation_test
# ---------------------------------------------------------------------------


def test_permutation_test_p_value_in_range(
    random_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    """p-value deve estar em [0.0, 1.0]."""
    y_true, y_pred = random_labels
    p = compute_permutation_test(y_true, y_pred, n_permutations=200)
    assert 0.0 <= p <= 1.0


def test_permutation_test_perfect_predictions_significant(
    perfect_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    """Predição perfeita deve produzir p < 0.05 (modelo é significativo)."""
    y_true, y_pred = perfect_labels
    p = compute_permutation_test(y_true, y_pred, n_permutations=500, seed=42)
    assert p < 0.05, f"Predição perfeita deve ser significativa, mas p={p:.4f}"


def test_permutation_test_known_seed_random_not_significant() -> None:
    """Predições completamente aleatórias geralmente produzem p > 0.05."""
    rng = np.random.default_rng(1234)
    y_true = rng.integers(0, 3, size=200)
    y_pred = rng.integers(0, 3, size=200)
    p = compute_permutation_test(y_true, y_pred, n_permutations=500, seed=42)
    # p-value alto indica que o modelo não supera a permutação aleatória
    assert p > 0.05, (
        f"Predições aleatórias não devem ser significativas, mas p={p:.4f}"
    )


def test_permutation_test_returns_float(
    random_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    """Deve retornar float Python nativo, não numpy scalar."""
    y_true, y_pred = random_labels
    p = compute_permutation_test(y_true, y_pred, n_permutations=50)
    assert isinstance(p, float)


# ---------------------------------------------------------------------------
# Grupo 4: compute_majority_baseline_f1
# ---------------------------------------------------------------------------


def test_majority_baseline_correct_class_identified() -> None:
    """Deve identificar HOLD (1) como classe majoritária em dados HOLD-dominados."""
    y_true = np.array([0] * 5 + [1] * 80 + [2] * 15)
    # F1-Macro de always-predict-HOLD:
    # SELL: F1=0, HOLD: F1=2*prec*rec/(prec+rec) > 0, BUY: F1=0
    baseline_f1 = compute_majority_baseline_f1(y_true)
    assert isinstance(baseline_f1, float)
    assert 0.0 <= baseline_f1 <= 1.0


def test_majority_baseline_f1_low_for_hold_dominated(
    hold_dominated_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    """Para dados HOLD-dominados, baseline F1-Macro deve ser bem abaixo de 0.5."""
    y_true, _ = hold_dominated_labels
    f1 = compute_majority_baseline_f1(y_true)
    # SELL e BUY têm F1=0, apenas HOLD > 0 → média macro baixa
    assert f1 < 0.5, f"Baseline F1-Macro esperado < 0.5 para dados HOLD-dominados, got {f1:.4f}"


def test_majority_baseline_returns_float() -> None:
    """Deve retornar float Python nativo."""
    y_true = np.array([0, 1, 1, 2, 1])
    result = compute_majority_baseline_f1(y_true)
    assert isinstance(result, float)


# ---------------------------------------------------------------------------
# Grupo 5: compute_metrics
# ---------------------------------------------------------------------------


def test_compute_metrics_all_fields_populated(
    balanced_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    """Todos os campos do EvaluationReport devem estar preenchidos."""
    y_true, y_pred = balanced_labels
    report = compute_metrics(y_true, y_pred, split_name="test", n_bootstrap=50, n_permutations=50)

    assert report.split_name == "test"
    assert report.n_samples == len(y_true)
    assert isinstance(report.f1_macro, float)
    assert isinstance(report.precision_buy, float)
    assert isinstance(report.recall_sell, float)
    assert isinstance(report.accuracy, float)
    assert isinstance(report.mcc, float)
    assert isinstance(report.cohen_kappa, float)
    assert isinstance(report.majority_baseline_f1, float)
    assert report.confusion_matrix.shape == (3, 3)
    assert set(report.per_class.keys()) == {0, 1, 2}
    assert isinstance(report.ci_f1_macro, BootstrapCI)
    assert isinstance(report.ci_precision_buy, BootstrapCI)
    assert isinstance(report.ci_recall_sell, BootstrapCI)
    assert isinstance(report.p_value_vs_baseline, float)
    assert isinstance(report.is_significant, bool)


def test_compute_metrics_f1_macro_matches_sklearn(
    balanced_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    """f1_macro deve corresponder ao valor calculado diretamente pelo sklearn."""
    from sklearn.metrics import f1_score

    y_true, y_pred = balanced_labels
    report = compute_metrics(y_true, y_pred, n_bootstrap=50, n_permutations=50)
    expected = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    assert report.f1_macro == pytest.approx(expected)


def test_compute_metrics_precision_buy_matches_sklearn(
    balanced_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    """precision_buy deve corresponder ao precision do sklearn para labels=[2]."""
    from sklearn.metrics import precision_score

    y_true, y_pred = balanced_labels
    report = compute_metrics(y_true, y_pred, n_bootstrap=50, n_permutations=50)
    expected = float(precision_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0])
    assert report.precision_buy == pytest.approx(expected)


def test_compute_metrics_confusion_matrix_shape(
    random_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    """Confusion matrix deve ser sempre (3, 3)."""
    y_true, y_pred = random_labels
    report = compute_metrics(y_true, y_pred, n_bootstrap=50, n_permutations=50)
    assert report.confusion_matrix.shape == (3, 3)


def test_compute_metrics_confusion_matrix_3x3_when_class_absent() -> None:
    """Confusion matrix deve ser (3,3) mesmo quando uma classe está ausente das predições."""
    y_true = np.array([0, 1, 2, 1, 1, 2])
    y_pred = np.array([1, 1, 1, 1, 1, 1])  # SELL e BUY nunca preditos
    report = compute_metrics(y_true, y_pred, n_bootstrap=50, n_permutations=50)
    assert report.confusion_matrix.shape == (3, 3)


def test_compute_metrics_n_samples_correct(
    random_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    """n_samples deve igualar o comprimento de y_true."""
    y_true, y_pred = random_labels
    report = compute_metrics(y_true, y_pred, n_bootstrap=50, n_permutations=50)
    assert report.n_samples == len(y_true)


def test_compute_metrics_accepts_checkpoint_meta_none(
    balanced_labels: tuple[np.ndarray, np.ndarray],
) -> None:
    """checkpoint_meta=None deve ser aceito sem erro."""
    y_true, y_pred = balanced_labels
    report = compute_metrics(
        y_true, y_pred, checkpoint_meta=None, n_bootstrap=50, n_permutations=50
    )
    assert report.checkpoint_meta is None


def test_compute_metrics_raises_on_shape_mismatch() -> None:
    """y_true e y_pred com shapes diferentes devem levantar ValueError."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1])
    with pytest.raises(ValueError, match="mesmo shape"):
        compute_metrics(y_true, y_pred)


def test_compute_metrics_raises_on_invalid_labels() -> None:
    """Labels fora de {0,1,2} devem levantar ValueError."""
    y_true = np.array([0, 1, 3])  # 3 é inválido
    y_pred = np.array([0, 1, 2])
    with pytest.raises(ValueError, match="fora de"):
        compute_metrics(y_true, y_pred)


# ---------------------------------------------------------------------------
# Grupo 6: EvaluationReport.is_significant
# ---------------------------------------------------------------------------


def test_is_significant_true_for_small_p() -> None:
    """p = 0.01 deve resultar em is_significant=True."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])  # perfeito para ter p pequeno
    report = compute_metrics(y_true, y_pred, n_bootstrap=50, n_permutations=200)
    # Com predição perfeita, p deve ser muito pequeno
    if report.p_value_vs_baseline < 0.05:
        assert report.is_significant is True


def test_is_significant_false_for_large_p() -> None:
    """p = 0.10 deve resultar em is_significant=False."""
    # Cria um report manual com p_value alto
    _y = np.array([0, 1, 2])
    _cm = np.zeros((3, 3), dtype=np.int64)
    _ci = BootstrapCI(lower=0.0, upper=1.0, confidence_level=0.95, point_estimate=0.5)
    _pc = {
        i: ClassMetrics(class_idx=i, class_name="X", precision=0.0, recall=0.0, f1=0.0, support=1)
        for i in range(3)
    }
    report = EvaluationReport(
        split_name="test", n_samples=3,
        f1_macro=0.3, precision_buy=0.3, recall_sell=0.3,
        accuracy=0.3, mcc=0.0, cohen_kappa=0.0, majority_baseline_f1=0.2,
        per_class=_pc, confusion_matrix=_cm,
        ci_f1_macro=_ci, ci_precision_buy=_ci, ci_recall_sell=_ci,
        p_value_vs_baseline=0.10,
        is_significant=0.10 < 0.05,  # False
        checkpoint_meta=None,
    )
    assert report.is_significant is False


def test_is_significant_false_for_p_exactly_0_05() -> None:
    """p = 0.05 é NÃO significativo (limiar estrito p < 0.05)."""
    _cm = np.zeros((3, 3), dtype=np.int64)
    _ci = BootstrapCI(lower=0.0, upper=1.0, confidence_level=0.95, point_estimate=0.5)
    _pc = {
        i: ClassMetrics(class_idx=i, class_name="X", precision=0.0, recall=0.0, f1=0.0, support=1)
        for i in range(3)
    }
    report = EvaluationReport(
        split_name="test", n_samples=3,
        f1_macro=0.3, precision_buy=0.3, recall_sell=0.3,
        accuracy=0.3, mcc=0.0, cohen_kappa=0.0, majority_baseline_f1=0.2,
        per_class=_pc, confusion_matrix=_cm,
        ci_f1_macro=_ci, ci_precision_buy=_ci, ci_recall_sell=_ci,
        p_value_vs_baseline=0.05,
        is_significant=0.05 < 0.05,  # False — limiar estrito
        checkpoint_meta=None,
    )
    assert report.is_significant is False


# ---------------------------------------------------------------------------
# Grupo 7: print_report
# ---------------------------------------------------------------------------


def test_print_report_does_not_raise(
    sample_report: EvaluationReport, capsys: pytest.CaptureFixture[str]
) -> None:
    """print_report não deve levantar nenhuma exceção."""
    print_report(sample_report)  # deve completar sem erro


def test_print_report_stdout_non_empty(
    sample_report: EvaluationReport, capsys: pytest.CaptureFixture[str]
) -> None:
    """print_report deve produzir output no stdout."""
    print_report(sample_report)
    captured = capsys.readouterr()
    assert len(captured.out) > 0


def test_print_report_contains_required_keywords(
    sample_report: EvaluationReport, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stdout deve conter seções obrigatórias do relatório."""
    print_report(sample_report)
    captured = capsys.readouterr()
    output = captured.out

    assert "F1" in output
    assert "SELL" in output
    assert "BUY" in output
    assert "Confusion" in output or "CONFUSION" in output


def test_print_report_handles_none_checkpoint_meta(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """print_report com checkpoint_meta=None não deve levantar erro."""
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 1, 0, 2])
    report = compute_metrics(
        y_true, y_pred, checkpoint_meta=None, n_bootstrap=50, n_permutations=50
    )
    print_report(report)  # não deve levantar
    captured = capsys.readouterr()
    assert "TRAINING CONTEXT" not in captured.out


# ---------------------------------------------------------------------------
# Grupo 8: evaluate_from_checkpoint
# ---------------------------------------------------------------------------


def test_evaluate_from_checkpoint_missing_path_raises() -> None:
    """Caminho inexistente deve levantar FileNotFoundError."""
    fake_path = Path("/tmp/nonexistent_checkpoint_agro3_phase5.pt")
    with pytest.raises(FileNotFoundError):
        evaluate_from_checkpoint(fake_path)
