"""Testes de ``ficha.select.chunking``: offsets, cobertura e sobreposição."""

from __future__ import annotations

import random
from itertools import pairwise

import pytest

from ficha.select.chunking import chunk_document, chunk_page, find_cut
from ficha.types import Document, Page


def _text(seed: int, n_sentences: int = 80) -> str:
    rng = random.Random(seed)
    words = ["model", "data", "method", "metric", "result", "evaluation", "sample", "study"]
    sentences = []
    for i in range(n_sentences):
        s = " ".join(rng.choice(words) for _ in range(rng.randint(5, 25))).capitalize() + "."
        sentences.append(s + ("\n\n" if i % 7 == 6 else " "))
    return "".join(sentences).strip()


PAGES = [Page(1, _text(1)), Page(2, _text(2, 30)), Page(3, "short page."), Page(4, "")]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: f"p{p.number}")
@pytest.mark.parametrize(("size", "overlap"), [(1200, 200), (300, 50), (500, 0), (80, 30)])
def test_offsets_coverage_and_overlap(page: Page, size: int, overlap: int) -> None:
    chunks = chunk_page(page, "a.pdf", size, overlap)
    text = page.text
    covered = [False] * len(text)
    for c in chunks:
        assert text[c.start : c.end] == c.text  # invariante de offset
        assert c.page == page.number and c.arquivo == "a.pdf"
        assert 0 < len(c.text) <= size
        assert c.text == c.text.strip()
        for i in range(c.start, c.end):
            covered[i] = True
    # cobertura: todo caractere não-branco está em algum chunk
    assert all(covered[i] for i, ch in enumerate(text) if not ch.isspace())
    # sobreposição: pelo menos ``overlap`` entre consecutivos, e sempre avançando
    for a, b in pairwise(chunks):
        assert b.start > a.start
        assert a.end - b.start >= overlap
        assert a.end - b.start <= overlap + overlap // 2 + 30  # não cresce sem limite


def test_prefers_sentence_boundaries() -> None:
    page = PAGES[0]
    chunks = chunk_page(page, "a.pdf", 1200, 200)
    assert len(chunks) > 3
    for c in chunks[:-1]:
        assert c.text.endswith(".")


def test_empty_and_short_pages() -> None:
    assert chunk_page(Page(1, ""), "a.pdf", 100, 10) == []
    assert chunk_page(Page(1, "   \n\n "), "a.pdf", 100, 10) == []
    [only] = chunk_page(Page(1, "  hello world.  "), "a.pdf", 100, 10)
    assert (only.start, only.end, only.text) == (2, 14, "hello world.")


def test_no_whitespace_text_is_hard_cut() -> None:
    page = Page(1, "x" * 250)
    chunks = chunk_page(page, "a.pdf", 100, 20)
    assert all(page.text[c.start : c.end] == c.text for c in chunks)
    assert chunks[-1].end == 250


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (100, 50), (100, -1)])
def test_invalid_params(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_page(Page(1, "abc"), "a.pdf", size, overlap)


def test_find_cut_prefers_paragraph_then_sentence() -> None:
    text = "One sentence. Another one here.\n\nNew paragraph continues without end"
    assert text[: find_cut(text, 5, 50)].endswith("here.")
    assert find_cut("abc def ghi", 1, 100) == len("abc def ghi")


def test_chunk_document_order_and_pages() -> None:
    doc = Document("a.pdf", tuple(PAGES))
    chunks = chunk_document(doc, 300, 50)
    keys = [(c.page, c.start) for c in chunks]
    assert keys == sorted(keys)
    assert {c.page for c in chunks} == {1, 2, 3}
    for c in chunks:
        assert doc.page(c.page).text[c.start : c.end] == c.text
