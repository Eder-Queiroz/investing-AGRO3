"""
Download idempotente do catálogo de PDFs da BrasilAgro (AGRO3).

Carrega o catálogo de config/pdf_catalog.yaml e baixa cada PDF para
data/raw/pdfs/, mantendo um manifest.json que rastreia o estado de
cada download (URL → ManifestEntry).

Idempotência: se o manifest indicar status='success' para uma URL,
o download é pulado na próxima execução.

Crash-safety: manifest.json é atualizado após CADA download bem-sucedido,
não apenas no final do loop.

Nota sobre api.mziq.com: os URLs podem exigir cookies/referrer para acesso
bulk. Se retornarem 403, registra status='failed' no manifest e continua.
Teste manual de alguns URLs é recomendado antes de rodar download_all().
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.config import load_pipeline_config
from src.utils.logger import get_logger

logger: logging.Logger = get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CATALOG_PATH = _PROJECT_ROOT / "config" / "pdf_catalog.yaml"


@dataclass
class PdfEntry:
    """Entrada do catálogo de PDFs."""

    label: str
    url: str | None
    category: str
    filename: str | None
    period: str | None = None
    year: int | None = None
    type: str | None = None
    notes: str | None = None


@dataclass
class ManifestEntry:
    """Registro de estado de um download no manifest.json."""

    url: str
    local_path: str
    label: str
    category: str
    period: str | None
    downloaded_at: str  # ISO 8601 UTC
    status: str  # "success" | "failed"
    size_bytes: int
    error: str | None = None


class PdfDownloader:
    """Downloader idempotente do catálogo de PDFs da AGRO3.

    Responsável por:
    - Carregar catálogo de config/pdf_catalog.yaml
    - Baixar PDFs ausentes para data/raw/pdfs/
    - Manter manifest.json atualizado após cada download
    - Pular downloads já concluídos (idempotência por URL)

    Exemplo de uso:
        downloader = PdfDownloader()
        manifest = downloader.download_all()
        print(f"Sucesso: {sum(1 for e in manifest.values() if e.status == 'success')}")
    """

    RATE_LIMIT_SECONDS: float = 0.5  # pausa entre downloads
    REQUEST_TIMEOUT_SECONDS: int = 30

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or load_pipeline_config()
        raw_output: str = cfg["data_ingestion"]["output"]["pdfs"]
        self._output_dir: Path = _PROJECT_ROOT / raw_output
        self._manifest_path: Path = self._output_dir / "manifest.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_all(
        self,
        force_redownload: bool = False,
    ) -> dict[str, ManifestEntry]:
        """Baixa todos os PDFs do catálogo que ainda não foram baixados.

        Args:
            force_redownload: Se True, ignora o manifest e baixa tudo novamente.

        Returns:
            Dict {url: ManifestEntry} com o estado final de cada entrada.
        """
        catalog = self.load_catalog()
        manifest = self.load_manifest()
        self._output_dir.mkdir(parents=True, exist_ok=True)

        valid_entries = [e for e in catalog if e.url and e.filename]
        skipped_entries = [e for e in catalog if not e.url or not e.filename]

        if skipped_entries:
            for entry in skipped_entries:
                logger.debug(f"Pulando entrada sem URL/filename: {entry.label}")

        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": "BrasilAgro-Research/1.0 (Academic Use)",
                "Accept": "application/pdf,*/*",
                "Referer": "https://ri.brasilagro.com.br",
            }
        )

        success_count = 0
        skip_count = 0
        fail_count = 0

        for entry in valid_entries:
            assert entry.url is not None  # satisfaz o type checker
            assert entry.filename is not None

            already_done = (
                not force_redownload
                and entry.url in manifest
                and manifest[entry.url].status == "success"
            )
            if already_done:
                logger.debug(f"Pulando já baixado: {entry.filename}")
                skip_count += 1
                continue

            result = self._download_single(entry, session)
            manifest[entry.url] = result
            self.save_manifest(manifest)  # persiste após CADA download

            if result.status == "success":
                success_count += 1
                logger.info(f"OK: {entry.filename} ({result.size_bytes:,} bytes)")
            else:
                fail_count += 1
                logger.warning(f"FALHOU: {entry.filename} — {result.error}")

            time.sleep(self.RATE_LIMIT_SECONDS)

        logger.info(
            f"Download concluído: {success_count} ok, {skip_count} pulados, "
            f"{fail_count} falhas de {len(valid_entries)} entradas válidas"
        )
        return manifest

    def load_catalog(self) -> list[PdfEntry]:
        """Carrega e achata o catálogo YAML em lista de PdfEntry.

        Itera sobre todas as chaves de nível superior (exceto 'metadata')
        e concatena as listas em uma única lista plana.

        Returns:
            Lista de PdfEntry com todos os documentos do catálogo.

        Raises:
            FileNotFoundError: Se config/pdf_catalog.yaml não existir.
        """
        import yaml  # importação local — yaml é dependência de pyyaml já instalado

        if not _CATALOG_PATH.exists():
            raise FileNotFoundError(f"Catálogo de PDFs não encontrado: {_CATALOG_PATH}")

        with _CATALOG_PATH.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        entries: list[PdfEntry] = []
        for section_key, section_value in raw.items():
            if section_key == "metadata":
                continue
            if not isinstance(section_value, list):
                continue
            for item in section_value:
                entries.append(
                    PdfEntry(
                        label=item.get("label", ""),
                        url=item.get("url"),
                        category=item.get("category", section_key),
                        filename=item.get("filename"),
                        period=item.get("period"),
                        year=item.get("year"),
                        type=item.get("type"),
                        notes=item.get("notes"),
                    )
                )

        logger.info(f"Catálogo carregado: {len(entries)} entradas")
        return entries

    def load_manifest(self) -> dict[str, ManifestEntry]:
        """Carrega o manifest.json se existir, ou retorna dict vazio.

        Returns:
            Dict {url: ManifestEntry} com histórico de downloads.
        """
        if not self._manifest_path.exists():
            return {}

        with self._manifest_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)

        return {url: ManifestEntry(**entry) for url, entry in raw.items()}

    def save_manifest(self, manifest: dict[str, ManifestEntry]) -> None:
        """Persiste o manifest.json no diretório de PDFs.

        Args:
            manifest: Dict {url: ManifestEntry} a ser salvo.
        """
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {url: asdict(entry) for url, entry in manifest.items()}
        with self._manifest_path.open("w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _download_single(
        self,
        entry: PdfEntry,
        session: requests.Session,
    ) -> ManifestEntry:
        """Tenta baixar um único PDF com retry automático.

        Erros HTTP (ex.: 403) são capturados e registrados no manifest
        como status='failed' sem propagar a exceção — o loop continua.

        Args:
            entry: Entrada do catálogo com URL e filename.
            session: requests.Session compartilhada com headers configurados.

        Returns:
            ManifestEntry com status='success' ou status='failed'.
        """
        assert entry.url is not None
        assert entry.filename is not None

        local_path = self._output_dir / entry.filename

        try:
            content = self._fetch_with_retry(entry.url, session)
            local_path.write_bytes(content)
            return ManifestEntry(
                url=entry.url,
                local_path=str(local_path),
                label=entry.label,
                category=entry.category,
                period=entry.period,
                downloaded_at=datetime.now(timezone.utc).isoformat(),
                status="success",
                size_bytes=len(content),
                error=None,
            )
        except Exception as exc:
            error_msg = str(exc)
            logger.debug(f"Falha no download de {entry.filename}: {error_msg}")
            return ManifestEntry(
                url=entry.url,
                local_path=str(local_path),
                label=entry.label,
                category=entry.category,
                period=entry.period,
                downloaded_at=datetime.now(timezone.utc).isoformat(),
                status="failed",
                size_bytes=0,
                error=error_msg,
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _fetch_with_retry(self, url: str, session: requests.Session) -> bytes:
        """Executa o GET HTTP com retry exponencial.

        Args:
            url: URL do PDF.
            session: Session HTTP com headers configurados.

        Returns:
            Conteúdo binário do PDF.

        Raises:
            requests.HTTPError: Se o servidor retornar status >= 400.
            requests.Timeout: Se o request exceder TIMEOUT segundos.
        """
        response = session.get(url, timeout=self.REQUEST_TIMEOUT_SECONDS, stream=False)
        response.raise_for_status()
        return response.content


# ---------------------------------------------------------------------------
# Execução standalone: uv run python -m src.data_ingestion.pdf_downloader
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Download de PDFs da AGRO3")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-baixa mesmo que já esteja no manifest",
    )
    args = parser.parse_args()

    downloader = PdfDownloader()
    manifest = downloader.download_all(force_redownload=args.force)

    ok = sum(1 for e in manifest.values() if e.status == "success")
    fail = sum(1 for e in manifest.values() if e.status == "failed")
    print(f"\nResultado: {ok} ok, {fail} falhas")
