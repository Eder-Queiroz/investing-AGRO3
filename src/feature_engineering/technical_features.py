"""
Features técnicas — transformações para estacionariedade.

Redes neurais falham com séries de preços não-estacionárias porque o modelo
aprende a escala absoluta em vez de padrões. Este módulo transforma todas as
séries de nível (preços, commodities, câmbio) em diferenças ou retornos, que
são integradas de ordem I(0) — estacionárias.

Transformações aplicadas:
    Preço AGRO3:   log-retornos para defasagens [1, 4, 13] semanas
    Volatilidade:  desvio padrão de log-retorno 1s anualizado (janela 4s)
    RSI:           Índice de Força Relativa via suavização de Wilder (EWM)
    Price-52w-high: ratio limitado (0, 1] — proxy de momentum relativo
    Volume:        z-score rolante (janela 4s) — captura atividade anormal
    Soja/Milho:    log-retorno 4 semanas (horizonte de tendência agrícola)
    USD/BRL:       log-retorno 4 semanas (risco cambial)
    CDI/Selic real: delta de 4 semanas (mudanças na política monetária)

Por que RSI via Wilder (EWM) e não rolling mean?
    O RSI original de Wilder usa uma média móvel exponencialmente ponderada
    com alpha = 1/period e adjust=False. Isso produz um smoothing assimétrico
    que enfatiza movimentos recentes — matematicamente distinto de uma
    rolling(period).mean() simples. Usar SMA produz RSI sistematicamente
    diferente dos valores publicados em qualquer plataforma financeira.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import load_pipeline_config
from src.utils.logger import get_logger
from src.utils.validators import validate_columns

logger: logging.Logger = get_logger(__name__)

_PRICE_REQUIRED = ["close_adj", "volume"]
_COMMODITY_REQUIRED = ["soy_price_usd", "corn_price_usd", "usd_brl"]
_MACRO_REQUIRED = ["cdi_rate", "selic_real"]

_ALL_REQUIRED = _PRICE_REQUIRED + _COMMODITY_REQUIRED + _MACRO_REQUIRED


class TechnicalFeatureBuilder:
    """Constrói features técnicas a partir de preços, volume e macro.

    Todas as transformações produzem séries estacionárias adequadas para
    input de MLP. O método transform() adiciona colunas ao DataFrame de
    entrada sem modificar as colunas existentes.

    Satisfaz o protocolo FeatureBuilder definido em pipeline.py.

    Exemplo de uso:
        builder = TechnicalFeatureBuilder()
        df_com_features = builder.transform(df_consolidado)
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Args:
            config: Dicionário de pipeline_config. Se None, carrega pipeline_config.yaml.
        """
        cfg = config or load_pipeline_config()
        fe_cfg = cfg["feature_engineering"]
        self._rsi_period: int = int(fe_cfg["rsi_period"])
        self._vol_window: int = int(fe_cfg["volatility_window_weeks"])
        self._momentum_windows: list[int] = list(fe_cfg["momentum_windows_weeks"])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona todas as features técnicas ao DataFrame de entrada.

        Não modifica colunas existentes. Retorna uma cópia com colunas novas.

        Args:
            df: DataFrame consolidado contendo pelo menos as colunas de
                preço, volume e macro listadas em _ALL_REQUIRED.

        Returns:
            DataFrame com as seguintes colunas adicionadas:
                log_return_1w, log_return_4w, log_return_13w
                volatility_4w
                rsi_14 (ou rsi_{period} conforme config)
                price_to_52w_high
                volume_zscore_4w
                log_return_soy_4w
                log_return_corn_4w
                log_return_usd_brl_4w
                delta_cdi_4w
                delta_selic_real_4w

        Raises:
            ValueError: Se qualquer coluna obrigatória estiver ausente.
        """
        validate_columns(df, _ALL_REQUIRED, context="TechnicalFeatureBuilder.transform")

        out = df.copy()

        # --- Log-retornos de preço ---
        log_ret_1w = self._log_return(out["close_adj"], lag=1)
        out["log_return_1w"] = log_ret_1w

        for lag in self._momentum_windows:
            col = f"log_return_{lag}w"
            if col not in out.columns:  # Evita sobrescrever se já existir
                out[col] = self._log_return(out["close_adj"], lag=lag)

        # --- Volatilidade realizada anualizada ---
        out["volatility_4w"] = self._annualized_volatility(log_ret_1w, self._vol_window)

        # --- RSI (Wilder) ---
        out[f"rsi_{self._rsi_period}"] = self._rsi(out["close_adj"], self._rsi_period)

        # --- Distância ao máximo de 52 semanas ---
        out["price_to_52w_high"] = self._price_to_52w_high(out["close_adj"])

        # --- Volume z-score ---
        out["volume_zscore_4w"] = self._volume_zscore(out["volume"], self._vol_window)

        # --- Commodities: log-retorno 4 semanas ---
        out["log_return_soy_4w"] = self._log_return(out["soy_price_usd"], lag=4)
        out["log_return_corn_4w"] = self._log_return(out["corn_price_usd"], lag=4)

        # --- Câmbio: log-retorno 4 semanas ---
        out["log_return_usd_brl_4w"] = self._log_return(out["usd_brl"], lag=4)

        # --- Política monetária: delta de taxas ---
        out["delta_cdi_4w"] = out["cdi_rate"] - out["cdi_rate"].shift(4)
        out["delta_selic_real_4w"] = out["selic_real"] - out["selic_real"].shift(4)

        new_cols = [c for c in out.columns if c not in df.columns]
        logger.debug(f"TechnicalFeatureBuilder: {len(new_cols)} colunas adicionadas.")
        return out

    # ------------------------------------------------------------------
    # Static helpers (reutilizáveis em testes unitários isolados)
    # ------------------------------------------------------------------

    @staticmethod
    def _log_return(series: pd.Series, lag: int) -> pd.Series:
        """Log-retorno para uma defasagem arbitrária.

        log(price[t] / price[t-lag])

        A série deve ser positiva (preços, índices). Não verifica isso
        explicitamente — valores zero ou negativos produzirão -inf ou NaN.

        Args:
            series: Série de preços ou índice (deve ser positiva).
            lag: Número de períodos de defasagem (semanas).

        Returns:
            Série de log-retornos. Primeiras 'lag' posições são NaN.
        """
        return np.log(series / series.shift(lag))

    @staticmethod
    def _annualized_volatility(log_return_1w: pd.Series, window: int) -> pd.Series:
        """Volatilidade realizada anualizada via desvio padrão rolante.

        vol[t] = std(log_return_1w[t-window+1..t]) × sqrt(52)

        A raiz de 52 converte volatilidade semanal em anualizada (assumindo
        52 semanas por ano de negociação).

        Args:
            log_return_1w: Série de log-retornos semanais.
            window: Tamanho da janela em semanas.

        Returns:
            Série de volatilidade anualizada. Primeiras window-1 posições NaN.
        """
        return log_return_1w.rolling(window=window, min_periods=window).std() * np.sqrt(52)

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:
        """RSI pelo método de Wilder (EWM com alpha=1/period, adjust=False).

        A suavização de Wilder é matematicamente equivalente a um EWM com
        alpha = 1/period e adjust=False. Isso é diferente de uma rolling mean
        simples — o Wilder dá mais peso a movimentos recentes de forma
        exponencialmente decrescente.

        Resultado sempre em [0, 100]. Valores próximos de 0 indicam
        oversold, próximos de 100 indicam overbought.

        Args:
            close: Série de preços de fechamento.
            period: Número de períodos (tipicamente 14).

        Returns:
            Série RSI com valores em [0, 100]. Primeiras posições NaN até
            acumulação suficiente de dados (min_periods=period via EWM).
        """
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        avg_gain = gain.ewm(
            alpha=1.0 / period, adjust=False, min_periods=period
        ).mean()
        avg_loss = loss.ewm(
            alpha=1.0 / period, adjust=False, min_periods=period
        ).mean()

        # Protege divisão por zero: quando avg_loss = 0, RS → inf → RSI → 100
        rs = avg_gain / avg_loss.replace(0.0, float("nan"))
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Quando avg_loss = 0 e avg_gain > 0: RSI deve ser 100
        rsi = rsi.where(avg_loss != 0.0, other=100.0)
        return rsi

    @staticmethod
    def _price_to_52w_high(close: pd.Series) -> pd.Series:
        """Ratio preço atual / máximo das últimas 52 semanas.

        Proxy de momentum: valores próximos de 1.0 indicam que o ativo
        está próximo de seu pico anual (força relativa alta). Valores
        baixos indicam distância do pico (possível fraqueza ou oportunidade).

        Sempre em (0, 1] por construção (close nunca supera o rolling max
        que inclui o próprio close).

        Args:
            close: Série de preços de fechamento.

        Returns:
            Série de ratios. Primeiras 51 posições NaN (janela incompleta).
        """
        rolling_max = close.rolling(window=52, min_periods=52).max()
        return close / rolling_max

    @staticmethod
    def _volume_zscore(volume: pd.Series, window: int) -> pd.Series:
        """Z-score do volume relativo à média e desvio padrão rolante.

        z[t] = (volume[t] - mean(volume[t-window+1..t])) / std(volume[t-window+1..t])

        Valores extremos (z > 2) indicam atividade de volume anormalmente
        elevada — potencial sinal de evento corporativo ou institucional.

        Args:
            volume: Série de volume semanal.
            window: Janela em semanas.

        Returns:
            Série de z-scores. NaN onde desvio padrão é zero (série constante)
            ou onde a janela está incompleta (primeiras window-1 posições).
        """
        rolling_mean = volume.rolling(window=window, min_periods=window).mean()
        rolling_std = volume.rolling(window=window, min_periods=window).std()
        # Protege divisão por zero: se todos os volumes forem iguais, std = 0
        safe_std = rolling_std.replace(0.0, float("nan"))
        return (volume - rolling_mean) / safe_std


# ---------------------------------------------------------------------------
# Entry point para smoke-test isolado
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys
    from pathlib import Path

    _PARQUET = (
        Path(__file__).resolve().parents[2] / "data" / "processed" / "features_weekly.parquet"
    )

    if not _PARQUET.exists():
        print(f"Parquet não encontrado: {_PARQUET}")
        sys.exit(1)

    _raw_cols = [
        "open_adj", "high_adj", "low_adj", "close_adj", "volume",
        "cdi_rate", "usd_brl", "selic_rate", "igpm", "ipca",
        "soy_price_usd", "corn_price_usd", "selic_real",
    ]
    _df = pd.read_parquet(_PARQUET, columns=_raw_cols)
    _builder = TechnicalFeatureBuilder()
    _result = _builder.transform(_df)

    _new = [c for c in _result.columns if c not in _raw_cols]
    print(f"\n{len(_new)} features técnicas adicionadas: {_new}")
    print(f"\nShape final: {_result.shape}")
    print(f"\nNaN por feature técnica:\n{_result[_new].isna().sum()}")
    print(f"\nEstatísticas:\n{_result[_new].describe().round(4)}")
    sys.exit(0)
