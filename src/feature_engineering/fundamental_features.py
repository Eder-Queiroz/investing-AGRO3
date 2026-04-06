"""
Features fundamentalistas — z-score expansivo sobre múltiplos de valuation.

Por que manter os ratios brutos?
    P/VP, EV/EBITDA, ROE, etc. são estacionários por construção — são ratios,
    não níveis absolutos. Uma MLP pode aprender o nível absoluto como sinal de
    valor (ex.: P/VP = 0.8 = desconto sobre patrimônio). Por isso, as colunas
    brutas são preservadas.

Por que adicionar z-scores expansivos?
    O nível absoluto de P/VP não informa se está caro ou barato *relativamente
    ao próprio histórico da empresa*. Um P/VP de 1.5 pode ser barato para um
    ativo que historicamente negocia a 3x, ou caro para um que negocia a 0.8x.
    O z-score expansivo normaliza o múltiplo pelo contexto histórico disponível
    até o ponto t, sem introduzir leakage futuro.

Por que expansivo (não rolante)?
    - Rolling window descartaria o histórico inicial, perdendo contexto de longo
      prazo crítico para value investing.
    - Full-sample z-score usaria dados futuros (leakage inaceitável).
    - Expanding window usa APENAS dados até t — causalmente correto.

min_periods = 8 (aproximadamente 2 anos de dados trimestrais):
    Garante que o z-score só é calculado quando há dados suficientes para
    uma estimativa estável de média e desvio padrão. Com < 8 observações,
    os primeiros z-scores seriam extremamente instáveis e enganosos.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.utils.logger import get_logger
from src.utils.validators import validate_columns

logger: logging.Logger = get_logger(__name__)

_FUNDAMENTAL_COLS: list[str] = [
    "p_vpa",
    "ev_ebitda",
    "roe",
    "net_debt_ebitda",
    "gross_margin",
    "dividend_yield",
]

_MIN_EXPANDING_PERIODS: int = 8


class FundamentalFeatureBuilder:
    """Constrói features fundamentalistas com z-score expansivo.

    Mantém os ratios brutos como estão (já são estacionários por construção
    — são ratios, não níveis de preço) e adiciona z-scores expansivos para
    cada ratio.

    O z-score expansivo permite ao modelo avaliar se o múltiplo atual está
    caro ou barato em relação ao histórico disponível até aquele momento,
    sem usar informação futura.

    Satisfaz o protocolo FeatureBuilder definido em pipeline.py.

    Exemplo de uso:
        builder = FundamentalFeatureBuilder()
        df_com_zscores = builder.transform(df_consolidado)
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona colunas de z-score expansivo para cada fundamental.

        Para cada coluna em _FUNDAMENTAL_COLS, adiciona uma coluna
        '{col}_zscore' com o z-score calculado pela janela expansiva até t.

        Os ratios brutos são preservados sem modificação. Linhas onde o
        ratio bruto é NaN (período anterior ao lag de 45 dias) produzem
        z-score NaN — correto, pois não há dado para normalizar.

        Args:
            df: DataFrame com as 6 colunas de fundamentos. Primeiras linhas
                podem ser NaN (lag de CVM de 45 dias — comportamento esperado).

        Returns:
            DataFrame com 6 novas colunas: p_vpa_zscore, ev_ebitda_zscore,
            roe_zscore, net_debt_ebitda_zscore, gross_margin_zscore,
            dividend_yield_zscore.

        Raises:
            ValueError: Se alguma coluna fundamental estiver ausente.
        """
        validate_columns(df, _FUNDAMENTAL_COLS, context="FundamentalFeatureBuilder.transform")

        out = df.copy()

        for col in _FUNDAMENTAL_COLS:
            zscore_col = f"{col}_zscore"
            out[zscore_col] = self._expanding_zscore(out[col], _MIN_EXPANDING_PERIODS)
            logger.debug(
                f"Z-score expansivo: '{col}' → '{zscore_col}' | "
                f"NaN: {int(out[zscore_col].isna().sum())} / {len(out)}"
            )

        new_cols = [f"{c}_zscore" for c in _FUNDAMENTAL_COLS]
        logger.debug(
            f"FundamentalFeatureBuilder: {len(new_cols)} colunas de z-score adicionadas."
        )
        return out

    @staticmethod
    def _expanding_zscore(series: pd.Series, min_periods: int) -> pd.Series:
        """Z-score expansivo (causal): usa apenas dados até t inclusive.

        z[t] = (x[t] - mean(x[0..t])) / std(x[0..t])

        Apenas observações não-NaN são consideradas na janela expansiva.
        Isso é correto para fundamentos que começam com NaN: o z-score
        começa a ser calculado a partir da primeira observação válida,
        sem contaminar com NaN.

        Args:
            series: Série de valores (pode conter NaN iniciais do lag de CVM).
            min_periods: Mínimo de observações válidas para produzir um valor.
                         Com menos de min_periods valores não-NaN, retorna NaN.

        Returns:
            Série de z-scores. NaN onde:
              - O valor original é NaN
              - Há menos de min_periods observações válidas até t
              - O desvio padrão expansivo é zero (série constante até t)
        """
        exp_mean = series.expanding(min_periods=min_periods).mean()
        exp_std = series.expanding(min_periods=min_periods).std()
        # Protege divisão por zero: série constante → std = 0 → z-score indefinido
        safe_std = exp_std.replace(0.0, float("nan"))
        return (series - exp_mean) / safe_std


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

    _df = pd.read_parquet(_PARQUET, columns=_FUNDAMENTAL_COLS)
    _builder = FundamentalFeatureBuilder()
    _result = _builder.transform(_df)

    _new = [f"{c}_zscore" for c in _FUNDAMENTAL_COLS]
    print(f"\n{len(_new)} z-scores adicionados: {_new}")
    print(f"\nShape final: {_result.shape}")
    print(f"\nNaN por z-score:\n{_result[_new].isna().sum()}")
    print(f"\nEstatísticas:\n{_result[_new].describe().round(4)}")
    sys.exit(0)
