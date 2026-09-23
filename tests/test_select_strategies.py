"""Testes das três estratégias da Seção 4.1 e do registro."""

from __future__ import annotations

import os
from itertools import pairwise

import numpy as np
import pytest

from ficha.config import get_settings
from ficha.ingest import load_pdf
from ficha.ingest.synthetic import generate_synthetic_corpus
from ficha.select import (
    DEFAULT_QUERIES,
    SELECTOR_NAMES,
    Embedder,
    FirstPagesSelector,
    HashingEmbedder,
    KeywordSectionSelector,
    SemanticSelector,
    SentenceTransformerEmbedder,
    build_selector,
)
from ficha.select.keyword import find_headings
from ficha.types import Context, ContextSelector, Document, Page


def _assert_valid_context(ctx: Context, doc: Document, strategy: str) -> None:
    assert ctx.strategy == strategy
    assert ctx.arquivo == doc.arquivo
    assert ctx.chunks, "contexto vazio"
    for c in ctx.chunks:
        page_text = doc.page(c.page).text
        assert c.text in page_text
        assert page_text[c.start : c.end] == c.text
    rendered = ctx.render()
    for page in ctx.pages:
        assert f"[p. {page}]" in rendered
    keys = [(c.page, c.start) for c in ctx.chunks]
    assert keys == sorted(keys), "chunks fora da ordem do artigo"


def _selectors() -> list[ContextSelector]:
    return [
        FirstPagesSelector(),
        KeywordSectionSelector(),
        SemanticSelector(HashingEmbedder()),
    ]


@pytest.fixture(scope="module")
def synthetic_docs(tmp_path_factory: pytest.TempPathFactory) -> list[Document]:
    out = tmp_path_factory.mktemp("synthetic_select")
    return [load_pdf(out / a.arquivo) for a in generate_synthetic_corpus(out, n=4, seed=11)]


# --------------------------------------------------------------------------- contrato comum


@pytest.mark.parametrize("selector", _selectors(), ids=lambda s: s.name)
def test_contract_on_fixture(selector: ContextSelector, doc: Document) -> None:
    assert isinstance(selector, ContextSelector)
    _assert_valid_context(selector.select(doc), doc, selector.name)


@pytest.mark.parametrize("selector", _selectors(), ids=lambda s: s.name)
def test_contract_on_synthetic_pdfs(
    selector: ContextSelector, synthetic_docs: list[Document]
) -> None:
    for d in synthetic_docs:
        _assert_valid_context(selector.select(d), d, selector.name)


# --------------------------------------------------------------------------- first_pages


def test_first_pages_takes_n_pages(doc: Document) -> None:
    ctx = FirstPagesSelector(n_pages=2).select(doc)
    assert ctx.pages == (1, 2)
    assert ctx.chunks[0].text == doc.page(1).text
    assert ctx.params["n_pages"] == 2


def test_first_pages_max_chars_truncates(doc: Document) -> None:
    ctx = FirstPagesSelector(n_pages=3, max_chars=300).select(doc)
    assert ctx.n_chars <= 300
    assert ctx.params["truncated"] is True


def test_first_pages_skips_empty_pages() -> None:
    d = Document("x.pdf", (Page(1, ""), Page(2, "Some text here."), Page(3, "More.")))
    ctx = FirstPagesSelector(n_pages=2).select(d)
    assert ctx.pages == (2,)


# --------------------------------------------------------------------------- keyword


def test_keyword_finds_methods_and_limitations(doc: Document) -> None:
    ctx = KeywordSectionSelector().select(doc)
    labels = " | ".join(c.label or "" for c in ctx.chunks)
    assert "3 Methods" in labels
    assert "5 Limitations" in labels
    assert "Abstract" in labels
    assert ctx.params["fallback"] is None
    lim = next(c for c in ctx.chunks if "Limitations" in (c.label or ""))
    assert lim.page == 3 and "single institution" in lim.text


def test_keyword_without_limitations(doc_sem_limitacao: Document) -> None:
    ctx = KeywordSectionSelector().select(doc_sem_limitacao)
    assert not any("Limitations" in (c.label or "") for c in ctx.chunks)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("3 Methods\n\nWe do X.", "methods"),
        ("3. Methodology\n\nWe do X.", "methods"),
        ("III. APPROACH\n\nWe do X.", "methods"),
        ("Materials and Methods\n\nWe do X.", "methods"),
        ("2.1 Dataset\n\nWe use Y.", "data"),
        ("Threats to Validity\n\nSome.", "limitations"),
        ("5 Limitations and Future Work\n\nSome.", "limitations"),
        ("Abstract. We study Z.", "abstract"),
        ("VI. EVALUATION\n\nSome.", "evaluation"),
        ("4 Experiments\n\nSome.", "experiments"),
    ],
)
def test_heading_regex_variants(text: str, expected: str) -> None:
    heads = find_headings(Document("x.pdf", (Page(1, text),)))
    assert [h.section for h in heads] == [expected]


@pytest.mark.parametrize(
    "text",
    [
        "Data come from a single institution.",  # frase comum, não cabeçalho
        "We report results in Table 2.",
        "the methods section is below",
    ],
)
def test_heading_regex_rejects_prose(text: str) -> None:
    assert find_headings(Document("x.pdf", (Page(1, text),))) == []


def test_keyword_ignores_references() -> None:
    d = Document(
        "x.pdf",
        (
            Page(1, "Abstract. We study things.\n\n2 Methods\n\nWe do things."),
            Page(2, "References\n\n12. Data mining for dummies. 2001."),
        ),
    )
    assert [h.section for h in find_headings(d)] == ["abstract", "methods"]


def test_keyword_fallback_to_first_pages() -> None:
    d = Document("x.pdf", tuple(Page(i, f"Plain prose on page {i}.") for i in range(1, 5)))
    ctx = KeywordSectionSelector().select(d)
    assert ctx.strategy == "keyword"
    assert ctx.params["fallback"] == "first_pages(2)"
    assert ctx.pages == (1, 2)


def test_keyword_respects_budget(synthetic_docs: list[Document]) -> None:
    sel = KeywordSectionSelector(window_chars=500, max_chars=1200)
    for d in synthetic_docs:
        assert sel.select(d).n_chars <= 1200


def test_keyword_window_continues_on_next_page() -> None:
    d = Document(
        "x.pdf",
        (
            Page(1, "Intro text.\n\n5 Limitations\n\nFirst part of the limitations."),
            Page(2, "Second part continues here.\n\n6 Conclusion\n\nDone."),
        ),
    )
    ctx = KeywordSectionSelector().select(d)
    labels = [c.label for c in ctx.chunks]
    assert "5 Limitations (cont.)" in " | ".join(x or "" for x in labels)
    cont = next(c for c in ctx.chunks if c.page == 2)
    assert cont.text.startswith("Second part continues here.")


def test_keyword_on_synthetic_pdfs_finds_sections(synthetic_docs: list[Document]) -> None:
    for d in synthetic_docs:
        sections = {h.section for h in find_headings(d)}
        assert {"abstract", "data", "methods", "results", "conclusion"} <= sections
        has_lim = "limitation" in d.full_text.lower()
        assert ("limitations" in sections) == has_lim


# --------------------------------------------------------------------------- semantic


def test_default_queries_cover_enunciado() -> None:
    assert "como o desempenho foi medido" in DEFAULT_QUERIES
    assert "o que este trabalho não consegue fazer" in DEFAULT_QUERIES


def test_semantic_limitation_query_hits_limitations_page(doc: Document) -> None:
    query = "limitações declaradas pelos autores e ameaças à validade do estudo"
    sel = SemanticSelector(HashingEmbedder(), queries=[query], top_k=1, chunk_size=300, overlap=50)
    [chunk] = sel.select(doc).chunks
    assert chunk.page == 3
    assert "Limitations" in chunk.text
    assert chunk.label == query
    assert chunk.score is not None and chunk.score > 0


def test_semantic_chunks_sorted_and_params(synthetic_docs: list[Document]) -> None:
    sel = SemanticSelector(HashingEmbedder(), top_k=2, chunk_size=600, overlap=100)
    for d in synthetic_docs:
        ctx = sel.select(d)
        keys = [(c.page, c.start) for c in ctx.chunks]
        assert keys == sorted(keys)
        assert all(c.label in DEFAULT_QUERIES for c in ctx.chunks)
        assert ctx.params["top_k"] == 2
        assert ctx.params["chunk_size"] == 600
        assert ctx.params["overlap"] == 100
        assert ctx.params["embedder"] == HashingEmbedder().name
        assert ctx.params["n_chunks_selected"] <= 2 * len(DEFAULT_QUERIES)


def test_semantic_merges_overlaps_without_duplicate_text(synthetic_docs: list[Document]) -> None:
    d = synthetic_docs[0]
    ctx = SemanticSelector(HashingEmbedder(), top_k=4, chunk_size=400, overlap=100).select(d)
    for a, b in pairwise(ctx.chunks):
        assert a.page != b.page or a.end < b.start


def test_semantic_max_chars(synthetic_docs: list[Document]) -> None:
    sel = SemanticSelector(HashingEmbedder(), top_k=6, max_chars=2500)
    for d in synthetic_docs:
        assert sel.select(d).n_chars <= 2500


def test_semantic_is_deterministic(doc: Document) -> None:
    a = SemanticSelector(HashingEmbedder(), top_k=2, chunk_size=300, overlap=50).select(doc)
    b = SemanticSelector(HashingEmbedder(), top_k=2, chunk_size=300, overlap=50).select(doc)
    assert a == b


# --------------------------------------------------------------------------- embedders


def test_hashing_embedder_normalized_and_deterministic() -> None:
    emb = HashingEmbedder(dim=64)
    assert isinstance(emb, Embedder)
    texts = ["limitations of the study", "limitações do estudo", ""]
    m = emb.embed(texts)
    assert m.shape == (3, 64)
    np.testing.assert_allclose(np.linalg.norm(m[:2], axis=1), 1.0, rtol=1e-5)
    assert np.linalg.norm(m[2]) == 0.0
    np.testing.assert_array_equal(m, emb.embed(texts))
    # n-gramas de caracteres aproximam cognatos PT↔EN
    other = emb.embed(["gradient boosting on hospital records"])[0]
    assert float(m[0] @ m[1]) > float(other @ m[1])


def test_sentence_transformer_embedder_is_lazy() -> None:
    emb = SentenceTransformerEmbedder("modelo/inexistente")
    assert isinstance(emb, Embedder)
    assert emb.name == "modelo/inexistente"
    assert emb._model is None  # nada carregado no construtor


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("FICHA_RUN_SLOW") != "1", reason="baixa modelo; rode com FICHA_RUN_SLOW=1"
)
def test_sentence_transformer_multilingual(doc: Document) -> None:  # pragma: no cover
    emb = SentenceTransformerEmbedder(get_settings().embedding_model)
    sel = SemanticSelector(
        emb, queries=["o que este trabalho não consegue fazer"], top_k=1, chunk_size=300, overlap=50
    )
    assert sel.select(doc).chunks[0].page == 3


# --------------------------------------------------------------------------- registro


@pytest.mark.parametrize("name", SELECTOR_NAMES)
def test_registry_builds_every_strategy(name: str, doc: Document) -> None:
    settings = get_settings()
    sel = build_selector(name, settings, embedder=HashingEmbedder())
    assert isinstance(sel, ContextSelector)
    assert sel.name == name
    _assert_valid_context(sel.select(doc), doc, name)


def test_registry_uses_settings() -> None:
    settings = get_settings(first_pages_n=2, chunk_size_chars=800, chunk_overlap_chars=100)
    fp = build_selector("first_pages", settings)
    assert isinstance(fp, FirstPagesSelector) and fp.n_pages == 2
    sem = build_selector("semantic", settings)
    assert isinstance(sem, SemanticSelector)
    assert (sem.chunk_size, sem.overlap, sem.top_k) == (800, 100, settings.semantic_top_k)
    assert isinstance(sem.embedder, SentenceTransformerEmbedder)
    assert sem.embedder.name == settings.embedding_model


def test_registry_unknown_name() -> None:
    with pytest.raises(ValueError, match="first_pages"):
        build_selector("tudo", get_settings())


def test_names_are_stable() -> None:
    assert SELECTOR_NAMES == ("first_pages", "keyword", "semantic")
