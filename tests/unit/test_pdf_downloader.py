"""Testes unitários para PdfDownloader.

Foco em idempotência, crash-safety e gestão do manifest.
Zero downloads reais.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.data_ingestion.pdf_downloader import ManifestEntry, PdfDownloader

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FAKE_PDF_BYTES = b"%PDF-1.4 fake content"

_CATALOG_YAML = textwrap.dedent(
    """
    metadata:
      ticker: "AGRO3.SA"

    releases:
      - label: "Release 4T25"
        url: "https://example.com/release_4T25.pdf"
        category: release
        period: "4T25"
        year: 2026
        filename: "release_4T25.pdf"

    fatos_relevantes:
      - label: "Fato Venda Fazenda X"
        url: "https://example.com/fr_venda_x.pdf"
        category: fato_relevante
        type: land_sale
        period: null
        year: 2024
        filename: "fr_venda_x.pdf"

      - label: "Fato sem URL"
        url: null
        category: fato_relevante
        type: corporate
        period: null
        year: null
        filename: null
        notes: "URL indisponível"
    """
)


def _make_downloader(tmp_path: Path) -> PdfDownloader:
    downloader = PdfDownloader.__new__(PdfDownloader)
    downloader._output_dir = tmp_path / "pdfs"
    downloader._manifest_path = downloader._output_dir / "manifest.json"
    downloader.RATE_LIMIT_SECONDS = 0.0  # sem sleep nos testes
    downloader.REQUEST_TIMEOUT_SECONDS = 30
    return downloader


def _write_catalog(tmp_path: Path, content: str) -> Path:
    catalog_path = tmp_path / "pdf_catalog.yaml"
    catalog_path.write_text(content, encoding="utf-8")
    return catalog_path


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------


class TestPdfDownloader:
    def test_load_catalog_flattens_all_sections(self, tmp_path: Path) -> None:
        catalog_path = _write_catalog(tmp_path, _CATALOG_YAML)
        downloader = _make_downloader(tmp_path)
        with patch("src.data_ingestion.pdf_downloader._CATALOG_PATH", catalog_path):
            entries = downloader.load_catalog()
        # 2 com URL + 1 sem URL = 3 entradas totais
        assert len(entries) == 3

    def test_load_catalog_parses_entry_fields(self, tmp_path: Path) -> None:
        catalog_path = _write_catalog(tmp_path, _CATALOG_YAML)
        downloader = _make_downloader(tmp_path)
        with patch("src.data_ingestion.pdf_downloader._CATALOG_PATH", catalog_path):
            entries = downloader.load_catalog()
        release = next(e for e in entries if e.category == "release")
        assert release.period == "4T25"
        assert release.year == 2026
        assert release.filename == "release_4T25.pdf"

    def test_download_all_happy_path(self, tmp_path: Path) -> None:
        catalog_path = _write_catalog(tmp_path, _CATALOG_YAML)
        downloader = _make_downloader(tmp_path)

        mock_response = MagicMock()
        mock_response.content = _FAKE_PDF_BYTES
        mock_response.raise_for_status.return_value = None
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        with (
            patch("src.data_ingestion.pdf_downloader._CATALOG_PATH", catalog_path),
            patch("src.data_ingestion.pdf_downloader.requests.Session", return_value=mock_session),
        ):
            manifest = downloader.download_all()

        success_entries = [e for e in manifest.values() if e.status == "success"]
        assert len(success_entries) == 2  # 2 entradas com URL válida

    def test_idempotent_skip_already_downloaded(self, tmp_path: Path) -> None:
        """Segunda execução não deve fazer nenhum download."""
        catalog_path = _write_catalog(tmp_path, _CATALOG_YAML)
        downloader = _make_downloader(tmp_path)
        downloader._output_dir.mkdir(parents=True)

        # Pre-populate manifest com status=success
        existing_manifest = {
            "https://example.com/release_4T25.pdf": ManifestEntry(
                url="https://example.com/release_4T25.pdf",
                local_path=str(downloader._output_dir / "release_4T25.pdf"),
                label="Release 4T25",
                category="release",
                period="4T25",
                downloaded_at="2026-04-01T00:00:00+00:00",
                status="success",
                size_bytes=100,
            ),
            "https://example.com/fr_venda_x.pdf": ManifestEntry(
                url="https://example.com/fr_venda_x.pdf",
                local_path=str(downloader._output_dir / "fr_venda_x.pdf"),
                label="Fato Venda Fazenda X",
                category="fato_relevante",
                period=None,
                downloaded_at="2026-04-01T00:00:00+00:00",
                status="success",
                size_bytes=200,
            ),
        }
        downloader.save_manifest(existing_manifest)

        mock_session = MagicMock()

        with (
            patch("src.data_ingestion.pdf_downloader._CATALOG_PATH", catalog_path),
            patch("src.data_ingestion.pdf_downloader.requests.Session", return_value=mock_session),
        ):
            downloader.download_all()

        # Session.get não deve ter sido chamado
        mock_session.get.assert_not_called()

    def test_force_redownload_ignores_manifest(self, tmp_path: Path) -> None:
        catalog_path = _write_catalog(tmp_path, _CATALOG_YAML)
        downloader = _make_downloader(tmp_path)
        downloader._output_dir.mkdir(parents=True)

        # Manifest diz success
        existing = {
            "https://example.com/release_4T25.pdf": ManifestEntry(
                url="https://example.com/release_4T25.pdf",
                local_path="x",
                label="x",
                category="release",
                period=None,
                downloaded_at="2026-04-01T00:00:00+00:00",
                status="success",
                size_bytes=100,
            ),
        }
        downloader.save_manifest(existing)

        mock_response = MagicMock()
        mock_response.content = _FAKE_PDF_BYTES
        mock_response.raise_for_status.return_value = None
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        with (
            patch("src.data_ingestion.pdf_downloader._CATALOG_PATH", catalog_path),
            patch("src.data_ingestion.pdf_downloader.requests.Session", return_value=mock_session),
        ):
            downloader.download_all(force_redownload=True)

        # Com force=True, deve ter baixado novamente
        assert mock_session.get.call_count == 2

    def test_manifest_persisted_after_each_download(self, tmp_path: Path) -> None:
        """Manifest deve ser salvo após CADA download (crash-safety)."""
        catalog_path = _write_catalog(tmp_path, _CATALOG_YAML)
        downloader = _make_downloader(tmp_path)

        save_calls: list[int] = []

        original_save = downloader.save_manifest

        def counting_save(manifest: dict) -> None:  # type: ignore[type-arg]
            save_calls.append(len(save_calls) + 1)
            original_save(manifest)

        mock_response = MagicMock()
        mock_response.content = _FAKE_PDF_BYTES
        mock_response.raise_for_status.return_value = None
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        with (
            patch("src.data_ingestion.pdf_downloader._CATALOG_PATH", catalog_path),
            patch("src.data_ingestion.pdf_downloader.requests.Session", return_value=mock_session),
            patch.object(downloader, "save_manifest", side_effect=counting_save),
        ):
            downloader.download_all()

        # 2 downloads válidos → 2 saves
        assert len(save_calls) == 2

    def test_manifest_key_is_url(self, tmp_path: Path) -> None:
        catalog_path = _write_catalog(tmp_path, _CATALOG_YAML)
        downloader = _make_downloader(tmp_path)

        mock_response = MagicMock()
        mock_response.content = _FAKE_PDF_BYTES
        mock_response.raise_for_status.return_value = None
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        with (
            patch("src.data_ingestion.pdf_downloader._CATALOG_PATH", catalog_path),
            patch("src.data_ingestion.pdf_downloader.requests.Session", return_value=mock_session),
        ):
            manifest = downloader.download_all()

        assert "https://example.com/release_4T25.pdf" in manifest
        assert "https://example.com/fr_venda_x.pdf" in manifest

    def test_http_error_recorded_as_failed_not_raised(self, tmp_path: Path) -> None:
        catalog_path = _write_catalog(tmp_path, _CATALOG_YAML)
        downloader = _make_downloader(tmp_path)

        import requests as req_lib

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req_lib.HTTPError("403 Forbidden")
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        with (
            patch("src.data_ingestion.pdf_downloader._CATALOG_PATH", catalog_path),
            patch("src.data_ingestion.pdf_downloader.requests.Session", return_value=mock_session),
        ):
            manifest = downloader.download_all()

        failed = [e for e in manifest.values() if e.status == "failed"]
        assert len(failed) == 2  # ambas as entradas válidas falharam

    def test_entries_without_url_skipped_silently(self, tmp_path: Path) -> None:
        catalog_path = _write_catalog(tmp_path, _CATALOG_YAML)
        downloader = _make_downloader(tmp_path)

        mock_response = MagicMock()
        mock_response.content = _FAKE_PDF_BYTES
        mock_response.raise_for_status.return_value = None
        mock_session = MagicMock()
        mock_session.get.return_value = mock_response

        with (
            patch("src.data_ingestion.pdf_downloader._CATALOG_PATH", catalog_path),
            patch("src.data_ingestion.pdf_downloader.requests.Session", return_value=mock_session),
        ):
            manifest = downloader.download_all()

        # Entrada sem URL não deve aparecer no manifest
        assert None not in manifest
        assert len(manifest) == 2

    def test_manifest_roundtrip_preserves_all_fields(self, tmp_path: Path) -> None:
        downloader = _make_downloader(tmp_path)
        downloader._output_dir.mkdir(parents=True)
        entry = ManifestEntry(
            url="https://example.com/test.pdf",
            local_path="/tmp/test.pdf",
            label="Test",
            category="release",
            period="1T24",
            downloaded_at="2026-04-01T12:00:00+00:00",
            status="success",
            size_bytes=1024,
            error=None,
        )
        manifest = {entry.url: entry}
        downloader.save_manifest(manifest)
        loaded = downloader.load_manifest()
        assert loaded["https://example.com/test.pdf"].size_bytes == 1024
        assert loaded["https://example.com/test.pdf"].status == "success"
