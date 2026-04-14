"""
HTTP client para Status Invest — dados fundamentalistas históricos da AGRO3.

Endpoints utilizados:
    DRE (Income Statement):
        GET /acao/getdre?code={code}&type=1&futureData=false&range.min={Y1}&range.max={Y2}
    Balanço Patrimonial (Ativos/Passivos):
        GET /acao/getativos?code={code}&type=1&futureData=false&range.min={Y1}&range.max={Y2}

type=1 → trimestral (quarterly). type=0 → anual.

Formato de resposta real (ambos endpoints):
    {
        "success": true,
        "data": {
            "grid": [
                {
                    "isHeader": true,
                    "row": 0,
                    "columns": [
                        {"value": "#"},
                        {"name": "DATA", "value": "4T2025"},
                        {"name": "AH",   "value": "AH"},
                        {"name": "AV",   "value": "AV"},
                        {"name": "DATA", "value": "3T2025"},
                        ...
                    ]
                },
                {
                    "isHeader": false,
                    "row": 1,
                    "columns": [...],
                    "gridLineModel": {
                        "key": "ReceitaLiquida",
                        "values": [183645000.00, 286643000.00, ...]
                    }
                },
                ...
            ]
        }
    }

Vantagem sobre yfinance:
    yfinance retorna apenas os últimos 4-5 trimestres para AGRO3.
    Status Invest cobre 2011-presente (~59 trimestres).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

_BASE_URL = "https://statusinvest.com.br"

_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://statusinvest.com.br/",
    "Accept": "application/json, text/plain, */*",
}

# Mapeamento: número do trimestre → mês de encerramento
_QUARTER_TO_MONTH: dict[int, int] = {1: 3, 2: 6, 3: 9, 4: 12}


class StatusInvestClient:
    """Cliente HTTP para a API de dados fundamentalistas do Status Invest.

    Responsabilidade única: buscar e parsear as respostas JSON dos endpoints
    de DRE e Balanço Patrimonial. Não aplica lags, alinhamentos temporais
    ou cálculo de ratios — isso é responsabilidade do FundamentalsFetcher.

    Exemplo:
        client = StatusInvestClient()
        dre = client.fetch_dre("AGRO3", start_year=2011, end_year=2025)
        bp  = client.fetch_balance("AGRO3", start_year=2011, end_year=2025)
    """

    def __init__(self) -> None:
        self._session: requests.Session = requests.Session()
        self._session.headers.update(_HEADERS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_dre(self, code: str, start_year: int, end_year: int) -> pd.DataFrame:
        """Busca o DRE (Demonstrativo de Resultado do Exercício) trimestral.

        Args:
            code: Ticker sem sufixo, ex: 'AGRO3'.
            start_year: Ano inicial do intervalo.
            end_year: Ano final do intervalo.

        Returns:
            DataFrame com DatetimeIndex de datas de fim de trimestre (ascending,
            tz-naive), colunas nomeadas pelo campo 'key' do gridLineModel
            (ex: 'ReceitaLiquida', 'Ebitda', 'MargemBruta', 'Roe', ...).

        Raises:
            requests.HTTPError: Se a requisição falhar após 3 tentativas.
            ValueError: Se a resposta não contiver a estrutura esperada.
        """
        url = (
            f"{_BASE_URL}/acao/getdre"
            f"?code={code.upper()}&type=1&futureData=false"
            f"&range.min={start_year}&range.max={end_year}"
        )
        logger.info(f"Buscando DRE: {code} de {start_year} a {end_year}")
        data = self._get(url)
        df = self._parse_grid(data)
        logger.debug(f"DRE: {len(df)} trimestres, colunas={df.columns.tolist()}")
        return df

    def fetch_dre_annual(self, code: str, start_year: int, end_year: int) -> pd.DataFrame:
        """Busca o DRE anual (type=0) — fornece ROE e indicadores anualizados.

        O endpoint anual retorna um período por ano com DatetimeIndex em 31/dez.
        Utilizado como fallback quando o DRE trimestral não contém ROE (caso AGRO3).

        Args:
            code: Ticker sem sufixo, ex: 'AGRO3'.
            start_year: Ano inicial do intervalo.
            end_year: Ano final do intervalo.

        Returns:
            DataFrame com DatetimeIndex de 31/dez de cada ano (ascending, tz-naive),
            colunas nomeadas pelo campo 'key' do gridLineModel
            (ex: 'ROE', 'MargemBruta', 'Ebitda', ...).

        Raises:
            requests.HTTPError: Se a requisição falhar após 3 tentativas.
            ValueError: Se a resposta não contiver a estrutura esperada.
        """
        url = (
            f"{_BASE_URL}/acao/getdre"
            f"?code={code.upper()}&type=0&futureData=false"
            f"&range.min={start_year}&range.max={end_year}"
        )
        logger.info(f"Buscando DRE anual: {code} de {start_year} a {end_year}")
        data = self._get(url)
        df = self._parse_annual_grid(data)
        logger.debug(f"DRE anual: {len(df)} anos, colunas={df.columns.tolist()}")
        return df

    def fetch_balance(self, code: str, start_year: int, end_year: int) -> pd.DataFrame:
        """Busca o Balanço Patrimonial trimestral via /acao/getativos.

        Args:
            code: Ticker sem sufixo, ex: 'AGRO3'.
            start_year: Ano inicial do intervalo.
            end_year: Ano final do intervalo.

        Returns:
            DataFrame com DatetimeIndex de datas de fim de trimestre (ascending,
            tz-naive), colunas nomeadas pelo campo 'key' do gridLineModel
            (ex: 'PatrimonioLiquidoConsolidado', 'CaixaeEquivalentesdeCaixa', ...).

        Raises:
            requests.HTTPError: Se a requisição falhar após 3 tentativas.
            ValueError: Se a resposta não contiver a estrutura esperada.
        """
        url = (
            f"{_BASE_URL}/acao/getativos"
            f"?code={code.upper()}&type=1&futureData=false"
            f"&range.min={start_year}&range.max={end_year}"
        )
        logger.info(f"Buscando Balanço Patrimonial: {code} de {start_year} a {end_year}")
        data = self._get(url)
        df = self._parse_grid(data)
        logger.debug(f"BP: {len(df)} trimestres, colunas={df.columns.tolist()}")
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _get(self, url: str) -> dict[str, Any]:
        """Executa GET com retry e retorna o JSON parseado.

        Args:
            url: URL completa do endpoint.

        Returns:
            Dict com o JSON da resposta (campo 'data').

        Raises:
            requests.HTTPError: Se o status code não for 2xx.
            ValueError: Se o JSON não contiver a estrutura 'data.grid' esperada.
        """
        logger.debug(f"GET {url}")
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        payload: dict[str, Any] = resp.json()
        if "data" not in payload or "grid" not in payload.get("data", {}):
            raise ValueError(
                f"Resposta inesperada de {url}: estrutura 'data.grid' ausente. "
                f"Chaves disponíveis: {list(payload.keys())}"
            )
        return payload["data"]

    def _parse_grid(self, data: dict[str, Any]) -> pd.DataFrame:
        """Converte o grid do Status Invest em DataFrame com DatetimeIndex.

        A resposta contém uma lista 'grid' onde:
        - A primeira linha (isHeader=True) contém os rótulos de trimestre
          nas colunas com name="DATA".
        - As demais linhas (isHeader=False) contêm 'gridLineModel.key' e
          'gridLineModel.values' com os valores numéricos.

        O resultado é ordenado ascendente (mais antigo primeiro) para
        alinhamento natural com o spine temporal do pipeline.

        Args:
            data: Dict com chave 'grid' (já extraído de payload["data"]).

        Returns:
            DataFrame ordenado ascendente por data de trimestre, com
            index.name='date'.
        """
        grid: list[dict[str, Any]] = data["grid"]

        # Extrai rótulos de trimestre do cabeçalho (isHeader=True)
        header_row = next(r for r in grid if r.get("isHeader", False))
        labels: list[str] = [
            col["value"]
            for col in header_row["columns"]
            if col.get("name") == "DATA"
        ]
        dates = self._parse_quarter_dates(labels)

        # Extrai linhas de dados (isHeader=False, com gridLineModel)
        rows: dict[str, list[float]] = {}
        for row in grid:
            if row.get("isHeader", False):
                continue
            glm = row.get("gridLineModel")
            if glm is None:
                continue
            key: str = glm["key"]
            values: list[float] = [
                float(v) if v is not None else np.nan for v in glm["values"]
            ]
            rows[key] = values

        df = pd.DataFrame(rows, index=dates)
        df.index.name = "date"
        return df.sort_index()

    def _parse_annual_grid(self, data: dict[str, Any]) -> pd.DataFrame:
        """Converte o grid anual do Status Invest em DataFrame com DatetimeIndex.

        A resposta anual usa 'data.years' (lista de anos inteiros) como índice
        e 'data.grid[i].gridLineModel' para os valores. Diferente do trimestral,
        não há linha de cabeçalho separada — o índice temporal vem de data.years.

        Os valores e os anos estão em ordem crescente (mais antigo primeiro).
        Cada ano é mapeado para 31/dez do respectivo ano.

        Args:
            data: Dict com chaves 'years' e 'grid' (extraído de payload["data"]).

        Returns:
            DataFrame com DatetimeIndex de 31/dez de cada ano, ordenado
            ascendente, com index.name='date'.
        """
        years: list[int] = data.get("years", [])
        if not years:
            raise ValueError("Resposta anual sem campo 'years'.")

        dates = pd.DatetimeIndex(
            [pd.Timestamp(year=y, month=12, day=31) for y in years]
        )

        rows: dict[str, list[float]] = {}
        for item in data.get("grid", []):
            if item.get("isHeader", False):
                continue
            glm = item.get("gridLineModel")
            if glm is None:
                continue
            key: str = glm["key"]
            raw_values: list = glm["values"]
            # Trunca ou expande para corresponder ao número de anos
            values: list[float] = [
                float(v) if v is not None else np.nan
                for v in raw_values[: len(years)]
            ]
            if len(values) < len(years):
                values += [np.nan] * (len(years) - len(values))
            rows[key] = values

        df = pd.DataFrame(rows, index=dates)
        df.index.name = "date"
        return df.sort_index()

    @staticmethod
    def _parse_quarter_dates(labels: list[str]) -> pd.DatetimeIndex:
        """Converte rótulos de trimestre em datas de fim de trimestre.

        Formato de entrada: '3T2024' (Q3 2024) → 2024-09-30.
        Suporta Q1-Q4 mapeados para 31/mar, 30/jun, 30/set, 31/dez.

        Args:
            labels: Lista de rótulos, ex: ['3T2024', '2T2024', '1T2024'].

        Returns:
            DatetimeIndex com as datas de final de cada trimestre.

        Raises:
            ValueError: Se um rótulo não estiver no formato esperado ('NTyyyy').
        """
        dates: list[pd.Timestamp] = []
        for label in labels:
            parts = label.split("T")
            if len(parts) != 2:
                raise ValueError(
                    f"Rótulo de trimestre inesperado: '{label}'. "
                    "Formato esperado: '3T2024'."
                )
            quarter = int(parts[0])
            year = int(parts[1])
            if quarter not in _QUARTER_TO_MONTH:
                raise ValueError(f"Trimestre inválido: {quarter}. Esperado 1-4.")
            month = _QUARTER_TO_MONTH[quarter]
            # MonthEnd(0): avança para o último dia do mês indicado
            dates.append(
                pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)
            )
        return pd.DatetimeIndex(dates)
