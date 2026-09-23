"""Testes de ``ficha.select.hybrid`` (semântica + seções de fechamento + pistas)."""

from __future__ import annotations

from itertools import pairwise

import pytest

from ficha.config import get_settings
from ficha.select import (
    HashingEmbedder,
    HybridSelector,
    KeywordSectionSelector,
    SemanticSelector,
    build_selector,
    limitation_sentence_recall,
)
from ficha.select.hybrid import tail_pages_before_references
from ficha.select.keyword import find_headings
from ficha.types import Context, Document, Page

PROBLEM_QUERY = "qual problema este trabalho tenta resolver"


def _semantic_first_page_only() -> SemanticSelector:
    # uma consulta, top-1, chunks pequenos: a semântica sozinha fica no abstract (p. 1)
    return SemanticSelector(
        HashingEmbedder(), queries=[PROBLEM_QUERY], top_k=1, chunk_size=300, overlap=50
    )


def _check(ctx: Context, doc: Document) -> None:
    assert ctx.strategy == "hybrid"
    for c in ctx.chunks:
        assert doc.page(c.page).text[c.start : c.end] == c.text
    keys = [(c.page, c.start) for c in ctx.chunks]
    assert keys == sorted(keys)
    for a, b in pairwise(ctx.chunks):
        assert a.page != b.page or a.end < b.start  # sem sobreposição


def test_semantic_alone_misses_limitations(doc: Document) -> None:
    ctx = _semantic_first_page_only().select(doc)
    assert 3 not in ctx.pages


def test_hybrid_adds_limitations_section(doc: Document) -> None:
    sel = HybridSelector(_semantic_first_page_only(), KeywordSectionSelector(), cue_windows=0)
    ctx = sel.select(doc)
    _check(ctx, doc)
    assert 3 in ctx.pages
    lim = next(c for c in ctx.chunks if c.page == 3)
    assert "5 Limitations" in lim.text and "limitations" in (lim.label or "")
    assert limitation_sentence_recall(doc, ctx).recall == 1.0
    fontes = ctx.params["chars_por_fonte"]
    assert isinstance(fontes, dict)
    assert fontes["limitations"] > 0 and fontes["semantic"] > 0
    assert sum(fontes.values()) == ctx.n_chars


def test_hybrid_closing_sections_only_no_abstract_by_keyword(doc: Document) -> None:
    sel = HybridSelector(_semantic_first_page_only(), KeywordSectionSelector(), cue_windows=0)
    labels = " ".join(c.label or "" for c in sel.select(doc).chunks)
    assert "closing: 6 Conclusion" in labels
    assert "Abstract" not in labels  # o abstract vem só pela semântica (sem rótulo de seção)
    assert "3 Methods" not in labels


def test_hybrid_cue_windows_find_limitations_without_heading() -> None:
    doc = Document(
        "x.pdf",
        (
            Page(1, "Abstract. We forecast solar flares with a convolutional network."),
            Page(
                2,
                "We train on ten years of magnetograms. One drawback of the approach is that "
                "it ignores the far side of the Sun. Training takes two days on one GPU.",
            ),
            Page(3, "The model reaches a TSS of 0.8 on the test set."),
        ),
    )
    sem = SemanticSelector(
        HashingEmbedder(), queries=["forecast solar flares convolutional network"], top_k=1
    )
    base = HybridSelector(sem, KeywordSectionSelector(), cue_windows=0)
    assert "drawback" not in base.select(doc).render()
    with_cues = HybridSelector(sem, KeywordSectionSelector())
    ctx = with_cues.select(doc)
    _check(ctx, doc)
    assert "One drawback of the approach" in ctx.render()
    assert any("pista: drawback" in (c.label or "") for c in ctx.chunks)


@pytest.mark.parametrize("budget", [150, 400, 900])
def test_hybrid_respects_budget(doc: Document, budget: int) -> None:
    sel = HybridSelector(
        SemanticSelector(HashingEmbedder(), top_k=3, chunk_size=300, overlap=50),
        KeywordSectionSelector(),
        max_chars=budget,
        tail_pages=2,
    )
    ctx = sel.select(doc)
    _check(ctx, doc)
    assert ctx.n_chars <= budget


def test_hybrid_priority_limitations_first_under_tight_budget(doc: Document) -> None:
    sel = HybridSelector(
        SemanticSelector(HashingEmbedder(), top_k=3, chunk_size=300, overlap=50),
        KeywordSectionSelector(),
        max_chars=200,
        cue_windows=0,
    )
    ctx = sel.select(doc)
    assert ctx.pages == (3,)
    assert "5 Limitations" in ctx.chunks[0].text


def test_hybrid_on_synthetic_pdfs(doc: Document, doc_sem_limitacao: Document) -> None:
    sel = HybridSelector(
        SemanticSelector(HashingEmbedder(), chunk_size=300, overlap=50), KeywordSectionSelector()
    )
    for d in (doc, doc_sem_limitacao):
        _check(sel.select(d), d)


def test_tail_pages_stop_at_references() -> None:
    doc = Document(
        "x.pdf",
        (
            Page(1, "Intro."),
            Page(2, "Body text."),
            Page(3, "Last body words.\n\nReferences\n\n[1] A. Author."),
            Page(4, "[2] B. Author."),
        ),
    )
    tail = tail_pages_before_references(doc, 2)
    assert [p.number for p in tail] == [2, 3]
    assert "References" not in tail[-1].text
    assert tail_pages_before_references(doc, 0) == []


def test_invalid_params() -> None:
    sem = _semantic_first_page_only()
    with pytest.raises(ValueError):
        HybridSelector(sem, KeywordSectionSelector(), tail_pages=-1)
    with pytest.raises(ValueError):
        HybridSelector(sem, KeywordSectionSelector(), cue_windows=-1)


def test_registry_builds_hybrid(doc: Document) -> None:
    sel = build_selector("hybrid", get_settings(), embedder=HashingEmbedder())
    assert isinstance(sel, HybridSelector)
    assert sel.name == "hybrid"
    assert sel.max_chars == getattr(get_settings(), "hybrid_max_chars", 12000)
    _check(sel.select(doc), doc)


# --------------------------------------------------------------------------- keyword (novos)


@pytest.mark.parametrize(
    ("text", "section"),
    [
        ("DISCUSSION AND CONCLUSIONS\n\nText.", "discussion"),
        ("4 Summary and Discussion\n\nText.", "discussion"),
        ("6. COMMENTS AND CONCLUSIONS\n\nText.", "conclusion"),
        ("Current limitations\n\nText.", "limitations"),
        ("Caveats\n\nText.", "limitations"),
        ("8 Outlook for the future\n\nText.", "outlook"),
        ("Future Work\n\nText.", "outlook"),
    ],
)
def test_new_closing_headings(text: str, section: str) -> None:
    heads = find_headings(Document("x.pdf", (Page(1, text),)))
    assert [h.section for h in heads] == [section]


def test_data_availability_is_not_data_section() -> None:
    heads = find_headings(Document("x.pdf", (Page(1, "Data Availability Statement\n\nX."),)))
    assert heads == []


def test_keyword_sections_param(doc: Document) -> None:
    ctx = KeywordSectionSelector(sections=["limitations"]).select(doc)
    assert ctx.pages == (3,)
    assert ctx.params["sections"] == ["limitations"]
    with pytest.raises(ValueError):
        KeywordSectionSelector(sections=["nope"])
