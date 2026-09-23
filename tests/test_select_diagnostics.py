"""Testes de ``ficha.select.diagnostics`` (recall offline das frases de limitação)."""

from __future__ import annotations

import pytest

from ficha.select import (
    LIMITATION_SENTENCE_RE,
    FirstPagesSelector,
    KeywordSectionSelector,
    limitation_sentence_recall,
    limitation_sentences,
    recall_table,
)
from ficha.select.diagnostics import RECALL_COLUMNS
from ficha.types import Chunk, Context, Document


@pytest.mark.parametrize(
    "sentence",
    [
        "Our study has limitations.",
        "One limitation of this approach is the sample size.",
        "The limitations of this study include a single site.",
        "The analysis is limited to active regions near disk center.",
        "We acknowledge that the sample is small.",
        "Testing each method is beyond the scope of this paper.",
        "We were not able to reproduce the baseline.",
        "These effects cannot be captured by the model.",
    ],
)
def test_regex_matches_declared_limitations(sentence: str) -> None:
    assert LIMITATION_SENTENCE_RE.search(sentence)


@pytest.mark.parametrize(
    "sentence",
    ["The model reaches an AUROC of 0.81.", "We use the MIMIC-IV database."],
)
def test_regex_ignores_plain_sentences(sentence: str) -> None:
    assert not LIMITATION_SENTENCE_RE.search(sentence)


def test_fixture_has_one_limitation_sentence_on_page_3(doc: Document) -> None:
    found = limitation_sentences(doc)
    assert [p for p, _ in found] == [3]
    assert found[0][1].startswith("Our study has limitations")


def test_recall_first_pages_vs_keyword(doc: Document) -> None:
    fp = limitation_sentence_recall(doc, FirstPagesSelector(n_pages=2).select(doc))
    assert (fp.n_sentences_doc, fp.n_sentences_in_context, fp.recall) == (1, 0, 0.0)
    assert fp.pages_doc == (3,)
    kw = limitation_sentence_recall(doc, KeywordSectionSelector().select(doc))
    assert kw.recall == 1.0 and kw.arquivo == doc.arquivo


def test_recall_is_none_without_sentences(doc_sem_limitacao: Document) -> None:
    ctx = FirstPagesSelector().select(doc_sem_limitacao)
    r = limitation_sentence_recall(doc_sem_limitacao, ctx)
    assert r.n_sentences_doc == 0 and r.recall is None and r.pages_doc == ()


def test_recall_tolerates_whitespace_differences(doc: Document) -> None:
    page = doc.page(3)
    start = page.text.index("5 Limitations")
    text = page.text[start:].replace(" ", "\n", 3)  # quebras onde havia espaço
    ctx = Context(doc.arquivo, "manual", (Chunk(doc.arquivo, 3, text, 0, len(text)),))
    assert limitation_sentence_recall(doc, ctx).recall == 1.0


def test_recall_table(docs: list[Document]) -> None:
    table = recall_table(
        docs, {"first_pages": FirstPagesSelector(n_pages=2), "keyword": KeywordSectionSelector()}
    )
    assert list(table.columns) == list(RECALL_COLUMNS)
    rows = table.set_index("estrategia")
    assert rows.loc["first_pages", "artigos_com_frases"] == 1
    assert rows.loc["first_pages", "artigos_cobertos"] == 0
    assert rows.loc["keyword", "artigos_cobertos"] == 1
    assert rows.loc["keyword", "frases_total"] == 1
    assert rows.loc["keyword", "recall_frases"] == 1.0
    assert rows.loc["keyword", "chars_medios"] > 0
