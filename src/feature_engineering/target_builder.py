"""
Construção do target Y — variável resposta do modelo MLP.

O target é um label triclasse que representa o desempenho de AGRO3 nos
52 períodos seguintes a cada observação, comparado ao CDI (custo de
oportunidade brasileiro) acumulado no mesmo período:

    margem[t] = retorno_total_52s[t] - cdi_acumulado_52s[t]

    label[t] = 1   (COMPRAR)  se margem > buy_threshold
    label[t] = 0   (AGUARDAR) se sell_threshold <= margem <= buy_threshold
    label[t] = -1  (VENDER)   se margem < sell_threshold

Por que o preço ajustado dispensa rastreamento de dividendos?
    O yfinance usa auto_adjust=True, o que retroativamente ajusta todos os
    preços históricos pelos proventos pagos. Assim, close_adj[t+52] / close_adj[t] - 1
    já captura o retorno total (price appreciation + dividendos reinvestidos).

Últimas 52 linhas:
    Não há janela futura completa → forward_return e forward_cdi são NaN
    → label é pd.NA. Essas linhas devem ser excluídas do treinamento
    (responsabilidade do Phase 3 / sliding_window.py).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import load_model_config
from src.utils.logger import get_logger
from src.utils.validators import validate_columns

logger: logging.Logger = get_logger(__name__)

_REQUIRED_COLS = ["close_adj", "cdi_rate"]


class TargetBuilder:
    """Constrói o target Y de 3 classes para o modelo MLP.

    A label em t reflete o retorno total dos horizon_weeks períodos SEGUINTES
    a t, comparado ao CDI acumulado no mesmo período. Isso é correto por design
    — o modelo aprende features em t para prever o que acontece após t.

    As últimas horizon_weeks linhas terão target pd.NA. Devem ser excluídas
    antes do treinamento (responsabilidade do Phase 3).

    Exemplo de uso:
        builder = TargetBuilder()
        target = builder.build(df)  # df deve conter 'close_adj' e 'cdi_rate'
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Args:
            config: Dicionário de model_config. Se None, carrega model_config.yaml.
        """
        cfg = config or load_model_config()
        target_cfg = cfg["target"]
        self._buy_threshold: float = float(target_cfg["buy_threshold_pp"]) / 100.0
        self._sell_threshold: float = float(target_cfg["sell_threshold_pp"]) / 100.0
        self._horizon: int = int(target_cfg["horizon_weeks"])

    def build(self, df: pd.DataFrame) -> pd.Series:
        """Computa o target Y para cada semana do DataFrame consolidado.

        Args:
            df: DataFrame consolidado contendo pelo menos 'close_adj' e 'cdi_rate',
                com DatetimeIndex W-FRI ordenado crescentemente.

        Returns:
            pd.Series com dtype pd.Int8Dtype(), name='target'. Valores são
            1 (COMPRAR), 0 (AGUARDAR), -1 (VENDER), ou pd.NA para as
            últimas horizon_weeks linhas sem janela futura completa.

        Raises:
            ValueError: Se 'close_adj' ou 'cdi_rate' não estiverem em df.columns.
        """
        validate_columns(df, _REQUIRED_COLS, context="TargetBuilder.build")

        forward_return = self._compute_forward_return(df["close_adj"])
        forward_cdi = self._compute_forward_cdi(df["cdi_rate"])
        margin = forward_return - forward_cdi

        target = self._assign_label(margin)

        n_valid = int(target.notna().sum())
        n_nan = int(target.isna().sum())
        logger.info(
            f"Target construído: {n_valid} amostras válidas, "
            f"{n_nan} com janela futura incompleta (pd.NA)."
        )
        if n_valid > 0:
            dist = target.dropna().value_counts().sort_index()
            logger.info(f"Distribuição do target: {dist.to_dict()}")

        return target

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_forward_return(self, close: pd.Series) -> pd.Series:
        """Retorno total forward de horizon_weeks semanas.

        Como auto_adjust=True, dividendos já estão embutidos no preço ajustado:
            forward_return[t] = close_adj[t + horizon] / close_adj[t] - 1

        As últimas horizon_weeks posições retornam NaN (sem dados futuros).

        Args:
            close: Série de preço ajustado de fechamento semanal.

        Returns:
            Série de retorno forward. Últimas horizon_weeks posições são NaN.
        """
        future_close = close.shift(-self._horizon)
        return (future_close / close) - 1.0

    def _compute_forward_cdi(self, cdi_rate: pd.Series) -> pd.Series:
        """CDI composto forward de horizon_weeks semanas.

        Fórmula:
            weekly_factor[t] = (1 + cdi_rate[t] / 100) ^ (1 / horizon)
            forward_cdi[t] = ∏(weekly_factor[t+1..t+horizon]) - 1

        Implementação via rolling product sobre a série deslocada:
            - shift(-1) posiciona a semana t+1 na posição t
            - rolling(horizon).apply(prod) acumula as horizon semanas seguintes

        As últimas horizon_weeks posições retornam NaN.

        Args:
            cdi_rate: Série de CDI anualizado em % (ex.: 13.75 = 13.75% a.a.).

        Returns:
            Série de retorno CDI composto forward. Últimas horizon_weeks posições NaN.
        """
        # log_wf[t] = log do fator semanal em t
        log_wf = np.log((1.0 + cdi_rate / 100.0) ** (1.0 / self._horizon))

        # log_cum[t] = Σ(i=0..t) log_wf[i]  (soma cumulativa)
        log_cum = log_wf.cumsum()

        # Soma dos logs futuros: Σ(i=t+1..t+horizon) log_wf[i]
        #   = log_cum[t+horizon] - log_cum[t]
        # log_cum.shift(-horizon)[t] = log_cum[t+horizon]
        # → NaN para t > n - horizon - 1, ou seja, últimas horizon linhas NaN ✓
        log_forward = log_cum.shift(-self._horizon) - log_cum
        forward_cdi: pd.Series = np.exp(log_forward) - 1.0
        return forward_cdi

    def _assign_label(self, margin: pd.Series) -> pd.Series:
        """Mapeia margem (retorno - CDI) para label triclasse.

        Usa operações pandas-nativas (não np.where) para preservar pd.NA
        nas posições onde a margem é NaN (janela futura incompleta).

        Args:
            margin: Série de margens. NaN onde janela futura incompleta.

        Returns:
            pd.Series com pd.Int8Dtype(), name='target'.
            Valores: 1 (COMPRAR), 0 (AGUARDAR), -1 (VENDER), pd.NA.
        """
        label = pd.Series(pd.NA, index=margin.index, dtype=pd.Int8Dtype(), name="target")

        valid = margin.notna()
        label[valid & (margin > self._buy_threshold)] = pd.array([1], dtype=pd.Int8Dtype())[0]
        label[valid & (margin < self._sell_threshold)] = pd.array([-1], dtype=pd.Int8Dtype())[0]
        # Posições válidas que não são compra nem venda → aguardar
        label[
            valid
            & (margin >= self._sell_threshold)
            & (margin <= self._buy_threshold)
        ] = pd.array([0], dtype=pd.Int8Dtype())[0]

        return label


# ---------------------------------------------------------------------------
# Entry point para smoke-test isolado
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys
    from pathlib import Path

    _FE_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
    _PARQUET = _FE_DIR / "features_weekly.parquet"

    if not _PARQUET.exists():
        print(f"Parquet não encontrado: {_PARQUET}")
        print("Execute o pipeline completo primeiro.")
        sys.exit(1)

    _df = pd.read_parquet(_PARQUET, columns=["close_adj", "cdi_rate"])
    _builder = TargetBuilder()
    _target = _builder.build(_df)

    print(f"\nShape: {len(_target)} semanas")
    print(f"dtype: {_target.dtype}")
    print(f"\nDistribuição:\n{_target.value_counts(dropna=False).sort_index()}")
    print(f"\nÚltimas 5 linhas:\n{_target.tail()}")
    sys.exit(0)
