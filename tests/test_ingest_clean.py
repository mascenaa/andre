"""Testes de ``ficha.ingest.clean``: cada regra isolada e o pipeline inteiro."""

from __future__ import annotations

import pytest

from ficha.ingest.clean import (
    build_vocabulary,
    clean_text,
    dehyphenate,
    is_page_number,
    normalize_unicode,
    rebuild_paragraphs,
    should_keep_hyphen,
    strip_page_numbers,
    strip_repeated_lines,
)

# --------------------------------------------------------------------------- hifenização


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("an exam-\nple here", "an example\nhere"),
        ("the evalua-\ntion.", "the evaluation."),
        ("highly accu-\nrate models", "highly accurate\nmodels"),
    ],
)
def test_dehyphenate_joins_typographic_breaks(raw: str, expected: str) -> None:
    assert dehyphenate(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a state-of-the-\nart model", "a state-of-the-art\nmodel"),
        ("the Bayes-\nOptimal rule", "the Bayes-Optimal\nrule"),
        ("Sentinel-\n2 imagery", "Sentinel-2\nimagery"),
        ("from 2019-\n2020 data", "from 2019-2020\ndata"),
        ("with self-\nattention layers", "with self-attention\nlayers"),
    ],
)
def test_dehyphenate_keeps_legit_hyphens(raw: str, expected: str) -> None:
    assert dehyphenate(raw) == expected


def test_dehyphenate_uses_document_vocabulary() -> None:
    # "fine-tuning" hifenizado em outro ponto do documento: o hífen é legítimo
    vocab = build_vocabulary(["we apply fine-tuning to the encoder"])
    assert dehyphenate("after fine-\ntuning the", vocab) == "after fine-tuning\nthe"
    # "multilingual" aparece junto no documento: a quebra é tipográfica
    vocab = build_vocabulary(["a multilingual encoder"])
    assert dehyphenate("a multi-\nlingual model", vocab) == "a multilingual\nmodel"


def test_should_keep_hyphen_rules_order() -> None:
    assert should_keep_hyphen("", "exam", "ple", set()) is False
    assert should_keep_hyphen("state-of-", "the", "art", set()) is True
    assert should_keep_hyphen("", "cross", "validation", set()) is True
    assert should_keep_hyphen("", "cross", "ing", set()) is False  # sufixo solto → junta


def test_dehyphenate_does_not_cross_paragraphs() -> None:
    assert dehyphenate("end of para-\n\nNext one") == "end of para-\n\nNext one"


# --------------------------------------------------------------------------- parágrafos


def test_rebuild_paragraphs_joins_single_breaks_and_keeps_double() -> None:
    raw = (
        "This is the first line of a paragraph that is long enough\n"
        "and this is its continuation line of similar length here.\n\n"
        "Second paragraph."
    )
    out = rebuild_paragraphs(raw)
    assert out == (
        "This is the first line of a paragraph that is long enough and this is its "
        "continuation line of similar length here.\n\nSecond paragraph."
    )


def test_rebuild_paragraphs_splits_heading_glued_to_text() -> None:
    raw = "Abstract\nWe study the problem of forecasting readmission using records from a hospital."
    assert rebuild_paragraphs(raw).split("\n\n")[0] == "Abstract"


# --------------------------------------------------------------------------- unicode / espaço


def test_ligatures_and_nfkc() -> None:
    assert normalize_unicode("eﬃcient ﬁne ﬂow") == "efficient fine flow"
    assert normalize_unicode("a b​c") == "a bc"


def test_soft_hyphen_at_end_of_line_joins() -> None:
    assert clean_text("repre­\nsentation learning") == "representation learning"


def test_multiple_spaces_collapsed() -> None:
    assert clean_text("too    many \t spaces") == "too many spaces"


# --------------------------------------------------------------------------- página


@pytest.mark.parametrize("line", ["12", "Page 3", "- 4 -", "3 of 10", " 7 "])
def test_is_page_number(line: str) -> None:
    assert is_page_number(line)


@pytest.mark.parametrize("line", ["3 Methods", "0.81", "Table 2", "2019 data"])
def test_is_not_page_number(line: str) -> None:
    assert not is_page_number(line)


def test_strip_page_numbers_only_at_edges() -> None:
    text = "7\nFirst line\n42\nLast line\n8"
    assert strip_page_numbers(text) == "First line\n42\nLast line"


def test_strip_repeated_lines_removes_header_and_footer() -> None:
    bodies = ["Alpha beta gamma.", "Delta epsilon zeta.", "Eta theta iota.", "Kappa lambda mu."]
    pages = [f"Journal of Things · Vol. 3\n{b}\n{i}" for i, b in enumerate(bodies, start=1)]
    out = strip_repeated_lines(pages)
    for i, (page, body) in enumerate(zip(out, bodies, strict=True), start=1):
        assert "Journal of Things" not in page
        assert body in page
        assert not page.rstrip().endswith(str(i))


def test_strip_repeated_lines_keeps_body_repetition_and_single_page() -> None:
    body = "\n".join(f"line {k} of body text that is unique" for k in range(10))
    middle = "Repeated sentence in the middle."
    pages = [f"Top {i}\n{body}\n{middle}\n{body}\nEnd {i}" for i in range(3)]
    out = strip_repeated_lines(pages)
    assert all("Repeated sentence in the middle." in p for p in out)
    assert strip_repeated_lines(["only one page\n1"]) == ["only one page\n1"]


# --------------------------------------------------------------------------- pipeline

MESSY = (
    "Journal X\n"
    "3 Methods\n"
    "We propose a new repre-\n"
    "sentation for eﬃcient   retrieval with a state-of-the-\n"
    "art encoder trained with self-\n"
    "attention and cross-validation.\n"
    "\n"
    "The second paragraph starts here and it is fairly long so\n"
    "that the line break inside it becomes a space.\n"
    "12\n"
)


def test_clean_text_end_to_end() -> None:
    out = clean_text(MESSY)
    assert "representation" in out
    assert "efficient retrieval" in out
    assert "state-of-the-art encoder" in out
    assert "self-attention" in out
    assert "\n12" not in out and not out.endswith("12")
    assert "fairly long so that the line break" in out
    assert "\n\n" in out  # parágrafo preservado
    assert "  " not in out


@pytest.mark.parametrize(
    "raw",
    [
        MESSY,
        "",
        "   \n\n  ",
        "single line",
        "exam-\nple\n\n\n\nnext\n3",
        "Abstract\nWe study X.\nShort.\nLonger line that goes on and on and on and on and on.",
    ],
)
def test_clean_text_is_idempotent(raw: str) -> None:
    once = clean_text(raw)
    assert clean_text(once) == once


def test_clean_text_is_deterministic() -> None:
    assert clean_text(MESSY) == clean_text(MESSY)
