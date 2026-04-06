"""
Pipeline de Feature Engineering — Fase 2.

Orquestra a sequência completa de transformações sobre os dados brutos da
Fase 1 e produz o DataFrame processado que alimentará o sliding window da
Fase 3.

Ordem de execução:
    1. DataConsolidator       → 19 colunas brutas (market + fundamentals + macro)
    2. TechnicalFeatureBuilder → +12 features técnicas (log-retornos, RSI, etc.)
    3. FundamentalFeatureBuilder → +6 z-scores expansivos de múltiplos
    4. TargetBuilder          → +1 coluna 'target' (pd.Int8Dtype, pd.NA nas últimas 52 linhas)
    5. _validate_output()     → assertions de integridade
    6. save()                 → data/processed/features_weekly.parquet

Extensibilidade (OCP):
    Para adicionar um novo grupo de features, basta:
    1. Criar uma classe que satisfaça o protocolo FeatureBuilder
       (qualquer classe com método transform(df) -> df)
    2. Instanciá-la e inseri-la em self._builders em __init__()
    Zero modificações no restante do código.

Shape esperado do output:
    ~989 linhas (semanas 2006-2025) × 38 colunas
    (19 brutas + 12 técnicas + 6 z-scores + 1 target)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from src.feature_engineering.consolidator import DataConsolidator
from src.feature_engineering.fundamental_features import FundamentalFeatureBuilder
from src.feature_engineering.target_builder import TargetBuilder
from src.feature_engineering.technical_features import TechnicalFeatureBuilder
from src.utils.config import load_pipeline_config
from src.utils.logger import get_logger
from src.utils.validators import validate_columns, validate_no_future_leakage

logger: logging.Logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Colunas que devem estar presentes no output final (excluindo 'target')
_EXPECTED_FEATURE_COLS: list[str] = [
    # Market (5)
    "open_adj",
    "high_adj",
    "low_adj",
    "close_adj",
    "volume",
    # Fundamentals brutos (6)
    "p_vpa",
    "ev_ebitda",
    "roe",
    "net_debt_ebitda",
    "gross_margin",
    "dividend_yield",
    # Macro (8)
    "cdi_rate",
    "usd_brl",
    "selic_rate",
    "igpm",
    "ipca",
    "soy_price_usd",
    "corn_price_usd",
    "selic_real",
    # Technical features (12)
    "log_return_1w",
    "log_return_4w",
    "log_return_13w",
    "volatility_4w",
    "rsi_14",
    "price_to_52w_high",
    "volume_zscore_4w",
    "log_return_soy_4w",
    "log_return_corn_4w",
    "log_return_usd_brl_4w",
    "delta_cdi_4w",
    "delta_selic_real_4w",
    # Fundamental z-scores (6)
    "p_vpa_zscore",
    "ev_ebitda_zscore",
    "roe_zscore",
    "net_debt_ebitda_zscore",
    "gross_margin_zscore",
    "dividend_yield_zscore",
]


class FeatureBuilder(Protocol):
    """Protocolo para construtores de features. Satisfaz OCP.

    Qualquer classe com método transform(df: pd.DataFrame) -> pd.DataFrame
    satisfaz este protocolo automaticamente via duck typing (typing.Protocol).
    Não é necessário herança explícita.

    Para adicionar um novo grupo de features ao pipeline:
        1. Crie uma classe com def transform(self, df: pd.DataFrame) -> pd.DataFrame
        2. Instancie-a e adicione-a à lista self._builders em FeatureEngineeringPipeline.__init__
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforma o DataFrame adicionando novas colunas de features.

        Args:
            df: DataFrame de entrada.

        Returns:
            DataFrame com novas colunas de features adicionadas.
            As colunas existentes devem ser preservadas sem modificação.
        """
        ...


class FeatureEngineeringPipeline:
    """Orquestra todo o pipeline de feature engineering da Fase 2.

    Recebe os dados brutos da Fase 1 via DataConsolidator (que chama
    fetcher.load() para cada fonte) e produz um único Parquet processado
    em data/processed/features_weekly.parquet.

    Exemplo de uso:
        pipeline = FeatureEngineeringPipeline()
        df = pipeline.run("2006-01-01", "2025-12-31")
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Args:
            config: Dicionário de pipeline_config. Se None, carrega pipeline_config.yaml.
        """
        cfg = config or load_pipeline_config()
        fe_cfg = cfg["feature_engineering"]
        self._output_path: Path = _PROJECT_ROOT / fe_cfg["processed_output"]

        self._consolidator = DataConsolidator(config=cfg)
        self._builders: list[FeatureBuilder] = [
            TechnicalFeatureBuilder(config=cfg),
            FundamentalFeatureBuilder(),
        ]
        self._target_builder = TargetBuilder()

    def run(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Executa o pipeline completo de feature engineering.

        Args:
            start_date: Data inicial no formato 'YYYY-MM-DD'.
            end_date: Data final no formato 'YYYY-MM-DD'.

        Returns:
            DataFrame com todas as features e coluna 'target'. As últimas
            horizon_weeks linhas terão target=pd.NA (sem janela futura).

        Raises:
            FileNotFoundError: Se parquets da Fase 1 não existirem.
            ValueError: Se validações de schema ou leakage falharem.
        """
        logger.info(f"=== Fase 2 — Feature Engineering: {start_date} → {end_date} ===")

        # 1. Consolidar dados da Fase 1
        logger.info("Passo 1/4: Consolidando dados brutos...")
        df = self._consolidator.consolidate(start_date, end_date)

        # 2. Aplicar builders em sequência
        for i, builder in enumerate(self._builders, start=2):
            builder_name = type(builder).__name__
            logger.info(f"Passo {i}/{len(self._builders) + 1}: {builder_name}...")
            df = builder.transform(df)

        # 3. Construir target Y
        step = len(self._builders) + 2
        logger.info(f"Passo {step}/{step}: TargetBuilder...")
        df["target"] = self._target_builder.build(df)

        # 4. Validar e salvar
        self._validate_output(df)
        self.save(df)

        logger.info(f"=== Pipeline concluído: {df.shape[0]} semanas × {df.shape[1]} colunas ===")
        return df

    def save(self, df: pd.DataFrame) -> None:
        """Salva o DataFrame final como Parquet comprimido com snappy.

        Cria os diretórios pai se necessário.

        Args:
            df: DataFrame com features e target prontos para salvar.
        """
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self._output_path, compression="snappy", index=True)
        logger.info(
            f"Parquet salvo: {self._output_path} ({self._output_path.stat().st_size:,} bytes)"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_output(self, df: pd.DataFrame) -> None:
        """Valida o DataFrame final antes de salvar.

        Verificações:
        1. Ordenação temporal e ausência de duplicatas
        2. Todas as colunas de features esperadas presentes
        3. Integridade do target: últimas horizon_weeks linhas são pd.NA
        4. Invariantes: RSI ∈ [0,100], price_to_52w_high ≤ 1.0
        5. Log de distribuição do target

        Args:
            df: DataFrame com features e target.

        Raises:
            ValueError: Se qualquer validação falhar.
        """
        validate_no_future_leakage(df, "index", context="FeatureEngineeringPipeline.validate")
        validate_columns(df, _EXPECTED_FEATURE_COLS + ["target"], context="pipeline.validate")

        # Invariantes estruturais
        if "rsi_14" in df.columns:
            valid_rsi = df["rsi_14"].dropna()
            if not valid_rsi.empty:
                if not (valid_rsi >= 0.0).all() or not (valid_rsi <= 100.0).all():
                    raise ValueError(
                        "[pipeline.validate] RSI fora dos limites [0, 100]. "
                        f"min={valid_rsi.min():.2f}, max={valid_rsi.max():.2f}"
                    )

        if "price_to_52w_high" in df.columns:
            valid_p52 = df["price_to_52w_high"].dropna()
            if not valid_p52.empty:
                if not (valid_p52 <= 1.0 + 1e-10).all():
                    raise ValueError(
                        "[pipeline.validate] price_to_52w_high > 1.0 detectado. "
                        f"max={valid_p52.max():.6f}"
                    )

        # Target
        if df["target"].dtype != pd.Int8Dtype():
            raise ValueError(
                f"[pipeline.validate] dtype do target deve ser pd.Int8Dtype(), "
                f"obtido: {df['target'].dtype}"
            )

        n_valid = int(df["target"].notna().sum())
        n_na = int(df["target"].isna().sum())

        if n_valid == 0:
            raise ValueError(
                "[pipeline.validate] Nenhuma amostra com target válido. "
                "Verifique o intervalo de datas e o horizon."
            )

        classes = set(df["target"].dropna().unique())
        if not classes.issubset({-1, 0, 1}):
            raise ValueError(
                f"[pipeline.validate] Target contém valores fora de {{-1, 0, 1}}: {classes}"
            )

        logger.info(f"Validação OK — {n_valid} amostras válidas, {n_na} com target pd.NA")
        dist = df["target"].value_counts(dropna=False).sort_index()
        logger.info(f"Distribuição do target:\n{dist.to_string()}")


# ---------------------------------------------------------------------------
# Entry point para execução direta
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import sys

    _cfg = load_pipeline_config()
    _di_cfg = _cfg["data_ingestion"]
    _end = _di_cfg["end_date"] or pd.Timestamp.today().strftime("%Y-%m-%d")

    _pipeline = FeatureEngineeringPipeline(config=_cfg)
    _result = _pipeline.run(start_date=_di_cfg["start_date"], end_date=_end)

    print(f"\n{'=' * 60}")
    print(f"Shape final:  {_result.shape}")
    print(f"\nColunas ({len(_result.columns)}):")
    for col in _result.columns:
        print(f"  {col}")
    print("\nDistribuição do target:")
    print(_result["target"].value_counts(dropna=False).sort_index().to_string())
    print("\nNaN por coluna (apenas com NaN):")
    _nan = _result.isna().sum()
    _nan = _nan[_nan > 0]
    print(_nan.to_string() if not _nan.empty else "  Nenhuma")
    print("\nAmostras mais recentes:")
    print(
        _result[["close_adj", "log_return_1w", "rsi_14", "p_vpa", "p_vpa_zscore", "target"]].tail()
    )
    sys.exit(0)
