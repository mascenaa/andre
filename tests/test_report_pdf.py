from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pymupdf
import pytest

from ficha.report.content import format_cell, sample_content, table_rows
from ficha.report.pdf import SECTION_TITLES, ReportTooLongError, build_report_pdf, count_pages


def _text(path: Path) -> str:
    with pymupdf.open(path) as pdf:
        return "\n".join(page.get_text() for page in pdf)


def _first_page(path: Path) -> str:
    with pymupdf.open(path) as pdf:
        return str(pdf[0].get_text())


def test_sample_report_fits_and_has_all_sections(tmp_path: Path) -> None:
    out = build_report_pdf(sample_content(), tmp_path / "AtividadeI_Teste.pdf")
    assert out.exists()
    assert count_pages(out) <= 3
    text = _text(out)
    for title in SECTION_TITLES:
        assert title in text, title
    # acentos preservados (fonte com encoding correto)
    assert "ção" in text and "ê" in text and "é" in text


def test_members_on_first_page(tmp_path: Path) -> None:
    content = sample_content()
    out = build_report_pdf(content, tmp_path / "r.pdf")
    first = _first_page(out)
    for name in content.integrantes:
        assert name in first


def test_page_footer(tmp_path: Path) -> None:
    out = build_report_pdf(sample_content(), tmp_path / "r.pdf")
    n = count_pages(out)
    assert f"página 1 de {n}" in _first_page(out)


def test_too_long_raises(tmp_path: Path) -> None:
    content = sample_content()
    longo = replace(content, estrategia_justificativa="Justificativa muito longa. " * 900)
    with pytest.raises(ReportTooLongError) as exc:
        build_report_pdf(longo, tmp_path / "r.pdf", max_pages=3)
    assert exc.value.n_pages > 3
    assert exc.value.max_pages == 3


def _tall_png(path: Path, ratio: float) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4, 4 * ratio))
    ax.plot([0, 1], [0, 1])
    fig.savefig(path, dpi=40)
    plt.close(fig)
    return path


def _n_images(path: Path) -> int:
    with pymupdf.open(path) as pdf:
        return sum(len(page.get_images()) for page in pdf)


def test_figure_height_is_capped(tmp_path: Path) -> None:
    from ficha.report.pdf import FIGURE_MAX_HEIGHT

    fig = _tall_png(tmp_path / "tall.png", 1.3)
    out = build_report_pdf(replace(sample_content(), figuras=[fig]), tmp_path / "r.pdf")
    with pymupdf.open(out) as pdf:
        rects = [r for page in pdf for img in page.get_images() for r in page.get_image_rects(img)]
    assert rects and all(r.height <= FIGURE_MAX_HEIGHT + 1 for r in rects)


def test_figures_included_when_they_fit(tmp_path: Path) -> None:
    fig = _tall_png(tmp_path / "wide.png", 0.2)
    content = replace(sample_content(), figuras=[fig, fig])
    out = build_report_pdf(content, tmp_path / "r.pdf")
    assert count_pages(out) <= 3
    assert _n_images(out) >= 1


def test_figures_dropped_when_they_do_not_fit(tmp_path: Path, monkeypatch) -> None:
    from ficha.report import pdf as pdf_mod

    base = build_report_pdf(sample_content(), tmp_path / "base.pdf")
    n_base = count_pages(base)
    # Sem o teto de altura, uma figura de ~23 cm força uma página extra.
    monkeypatch.setattr(pdf_mod, "FIGURE_MAX_HEIGHT", 24 * 28.35)
    fig = _tall_png(tmp_path / "tall.png", 1.3)
    content = replace(sample_content(), figuras=[fig])
    out = build_report_pdf(content, tmp_path / "r.pdf", max_pages=n_base)
    assert count_pages(out) == n_base
    assert _n_images(out) == 0


def test_empty_nao_defensaveis(tmp_path: Path) -> None:
    content = replace(sample_content(), fichas_nao_defensaveis=[])
    out = build_report_pdf(content, tmp_path / "r.pdf")
    assert "Nenhuma ficha" in _text(out)


def test_requires_members() -> None:
    with pytest.raises(ValueError):
        replace(sample_content(), integrantes=[" "])


def test_table_rows_accepts_frame_and_dicts() -> None:
    frame = pd.DataFrame({"v": [0.5, 1234.5]}, index=pd.Index(["a", "b"], name="metrica"))
    headers, rows = table_rows(frame)
    assert headers == ["metrica", "v"]
    assert rows == [["a", "0,5"], ["b", "1.234,50"]]
    headers, rows = table_rows([{"x": 1, "y": None}, {"x": 2000, "z": True}])
    assert headers == ["x", "y", "z"]
    assert rows == [["1", "", ""], ["2.000", "", "sim"]]
    assert format_cell(float("nan")) == ""
    assert format_cell(7.0) == "7"
    assert format_cell(["a", "b"]) == "a; b"
