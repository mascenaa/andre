"""Testes de ``ficha.ingest.pdf``, ``ficha.ingest.cache`` e do gerador sintético."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from reportlab.pdfgen.canvas import Canvas

from ficha.ingest import (
    EmptyDocumentError,
    PdfLoadError,
    clean_text,
    load_corpus,
    load_document,
    load_or_ingest,
    load_pdf,
    save_document,
)
from ficha.ingest.synthetic import (
    MANIFEST_NAME,
    SyntheticArticle,
    generate_synthetic_corpus,
    make_synthetic_corpus,
)

Corpus = tuple[Path, list[SyntheticArticle]]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[SyntheticArticle]]:
    out = tmp_path_factory.mktemp("synthetic")
    return out, generate_synthetic_corpus(out, n=4, seed=7)


# --------------------------------------------------------------------------- gerador


def test_make_synthetic_corpus_is_deterministic(tmp_path: Path) -> None:
    a = make_synthetic_corpus(tmp_path / "a", n=2, seed=3)
    b = make_synthetic_corpus(tmp_path / "b", n=2, seed=3)
    assert [p.name for p in a] == ["artigo_01.pdf", "artigo_02.pdf"]
    for pa, pb in zip(a, b, strict=True):
        assert pa.read_bytes() == pb.read_bytes()


def test_manifest_is_ground_truth(corpus: tuple[Path, list[SyntheticArticle]]) -> None:
    out, articles = corpus
    manifest = json.loads((out / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert [a["arquivo"] for a in manifest["articles"]] == [a.arquivo for a in articles]
    assert [a.has_limitations for a in articles] == [True, False, True, False]
    assert all(3 <= a.n_pages <= 5 for a in articles)
    assert all(a.hyphenated_words for a in articles)


# --------------------------------------------------------------------------- load_pdf


def test_load_pdf_pages_and_metadata(corpus: tuple[Path, list[SyntheticArticle]]) -> None:
    out, articles = corpus
    for art in articles:
        doc = load_pdf(out / art.arquivo)
        assert doc.arquivo == art.arquivo
        assert doc.n_pages == art.n_pages
        assert [p.number for p in doc.pages] == list(range(1, art.n_pages + 1))
        assert doc.metadata["n_pages"] == art.n_pages
        assert doc.metadata["title"] == art.title
        assert doc.metadata["n_chars_clean"] == doc.n_chars
        assert doc.metadata["n_chars_raw"] > 0


def test_load_pdf_contains_sections(corpus: tuple[Path, list[SyntheticArticle]]) -> None:
    out, articles = corpus
    for art in articles:
        text = load_pdf(out / art.arquivo).full_text
        assert art.fields["dados"] in text
        assert art.fields["metodo"] in text
        assert "Abstract" in text
        if art.has_limitations:
            assert art.fields["limitacao"] in text
        else:
            assert "limitation" not in text.lower()


def test_load_pdf_removes_repeated_header_and_page_numbers(
    corpus: tuple[Path, list[SyntheticArticle]],
) -> None:
    out, articles = corpus
    for art in articles:
        doc = load_pdf(out / art.arquivo)
        assert "Synthetic Journal of Applied Data Science" not in doc.full_text
        for page in doc.pages:
            last_line = page.text.rsplit("\n", 1)[-1].strip()
            assert last_line != str(page.number)


def test_load_pdf_resolves_end_of_line_hyphens(
    corpus: tuple[Path, list[SyntheticArticle]],
) -> None:
    out, articles = corpus
    for art in articles:
        doc = load_pdf(out / art.arquivo)
        for page in doc.pages:
            assert "-\n" not in page.text
            # clean_text é idempotente também sobre o texto real extraído
            assert clean_text(page.text) == page.text
        # palavras quebradas dentro da página saem inteiras (a quebra entre páginas não é
        # desfeita de propósito: juntar mudaria a página da evidência)
        full = doc.full_text
        joined = [w for w in art.hyphenated_words if w in full]
        assert len(joined) >= 0.8 * len(art.hyphenated_words)
        for w in art.compound_breaks:
            assert w in full


def test_load_corpus_sorted(corpus: tuple[Path, list[SyntheticArticle]]) -> None:
    out, articles = corpus
    docs = load_corpus(out)
    assert [d.arquivo for d in docs] == sorted(a.arquivo for a in articles)


def test_load_corpus_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nope")


# --------------------------------------------------------------------------- erros


def _blank_pdf(path: Path, n_pages: int = 2) -> Path:
    c = Canvas(str(path))
    for _ in range(n_pages):
        c.rect(100, 100, 200, 200)  # só desenho, nenhum texto (como um escaneado)
        c.showPage()
    c.save()
    return path


def test_scanned_pdf_raises_empty_document(tmp_path: Path) -> None:
    pdf = _blank_pdf(tmp_path / "escaneado.pdf")
    with pytest.raises(EmptyDocumentError, match="OCR"):
        load_pdf(pdf)


def test_load_corpus_skip_errors(
    tmp_path: Path, corpus: tuple[Path, list[SyntheticArticle]]
) -> None:
    out, articles = corpus
    (tmp_path / articles[0].arquivo).write_bytes((out / articles[0].arquivo).read_bytes())
    _blank_pdf(tmp_path / "zz_escaneado.pdf")
    with pytest.raises(EmptyDocumentError):
        load_corpus(tmp_path)
    assert [d.arquivo for d in load_corpus(tmp_path, skip_errors=True)] == [articles[0].arquivo]


def test_not_a_pdf(tmp_path: Path) -> None:
    bogus = tmp_path / "bogus.pdf"
    bogus.write_text("isto não é um PDF")
    with pytest.raises(PdfLoadError):
        load_pdf(bogus)


def test_missing_pdf(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "missing.pdf")


# --------------------------------------------------------------------------- cache


def test_cache_roundtrip_and_sha(tmp_path: Path, corpus: Corpus) -> None:
    out, articles = corpus
    doc = load_pdf(out / articles[0].arquivo)
    path = save_document(doc, tmp_path / "processed")
    assert path.name == f"{doc.arquivo}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["sha256"] == doc.sha256()
    again = load_document(path)
    assert again == doc


def test_cache_detects_tampering(tmp_path: Path, corpus: Corpus) -> None:
    out, articles = corpus
    path = save_document(load_pdf(out / articles[0].arquivo), tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["pages"][0]["text"] += " adulterado"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="sha256"):
        load_document(path)


def test_load_or_ingest_uses_cache(
    tmp_path: Path, corpus: tuple[Path, list[SyntheticArticle]], monkeypatch: pytest.MonkeyPatch
) -> None:
    out, articles = corpus
    pdf = out / articles[1].arquivo
    processed = tmp_path / "processed"
    first = load_or_ingest(pdf, processed)
    assert (processed / f"{pdf.name}.json").is_file()

    import ficha.ingest.cache as cache_mod

    def _boom(_: Path) -> None:
        raise AssertionError("não deveria reprocessar")

    monkeypatch.setattr(cache_mod, "load_pdf", _boom)
    assert load_or_ingest(pdf, processed) == first
    with pytest.raises(AssertionError):
        load_or_ingest(pdf, processed, force=True)
