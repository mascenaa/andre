from __future__ import annotations

import re

from ficha.prompts.examples import FEW_SHOT_EXAMPLES

_PAGE_RE = re.compile(r"^\[p\. (\d+)\]$", re.MULTILINE)


def _page_of(context_text: str, trecho: str) -> int:
    pos = context_text.index(trecho)
    marks = [m for m in _PAGE_RE.finditer(context_text) if m.start() < pos]
    return int(marks[-1].group(1))


def test_pelo_menos_dois_exemplos_um_com_null():
    assert len(FEW_SHOT_EXAMPLES) >= 2
    assert any(ex.output.limitacao is None for ex in FEW_SHOT_EXAMPLES)
    assert any(ex.output.limitacao is not None for ex in FEW_SHOT_EXAMPLES)


def test_trecho_e_copia_literal_e_pagina_bate_com_marcador():
    for ex in FEW_SHOT_EXAMPLES:
        trecho = ex.output.evidencia.trecho
        assert trecho in ex.context_text, ex.name
        assert _page_of(ex.context_text, trecho) == ex.output.evidencia.pagina, ex.name


def test_formato_igual_ao_context_render():
    for ex in FEW_SHOT_EXAMPLES:
        assert ex.context_text.startswith("[p. ")
        assert "\n\n[p. " in ex.context_text


def test_exemplo_sem_limitacao_de_fato_nao_declara():
    ex = next(e for e in FEW_SHOT_EXAMPLES if e.output.limitacao is None)
    assert "limitation" not in ex.context_text.lower()
    assert "não informado" in ex.output.dados


def test_exemplo_com_limitacao_declara_no_texto():
    ex = next(e for e in FEW_SHOT_EXAMPLES if e.output.limitacao is not None)
    assert "limitation" in ex.context_text.lower()
