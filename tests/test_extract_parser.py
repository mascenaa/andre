from __future__ import annotations

import json

import pytest

from ficha.extract.parser import (
    REPAIR_COT_NO_ENVELOPE,
    REPAIR_EXTRACT_OBJECT,
    REPAIR_FENCES,
    REPAIR_LIMITACAO_MISSING,
    REPAIR_LIMITACAO_SENTINEL,
    REPAIR_PAGINA_INT,
    REPAIR_SMART_QUOTES,
    REPAIR_TRAILING_COMMA,
    REPAIR_UNWRAP,
    extract_first_object,
    parse_completion,
    remove_trailing_commas,
)
from ficha.types import ParseStatus

TRECHO = "The model reaches an AUROC of 0.81 on the held-out test set."


def _ficha(**over):
    d = {
        "problema": "Prever readmissão hospitalar.",
        "dados": "MIMIC-IV, 2008 a 2019.",
        "metodo": "LightGBM.",
        "metrica": "AUROC.",
        "limitacao": None,
        "evidencia": {"trecho": TRECHO, "pagina": 3},
    }
    d.update(over)
    return d


def _json(obj, **kw):
    return json.dumps(obj, ensure_ascii=False, indent=2, **kw)


def test_valido_ok():
    r = parse_completion(_json(_ficha()))
    assert r.status is ParseStatus.OK
    assert r.json_valid_first_try is True
    assert r.repairs == []
    assert r.error is None
    assert r.ficha is not None and r.ficha.evidencia.pagina == 3


def test_limitacao_string_ok():
    r = parse_completion(_json(_ficha(limitacao="Uma instituição só.")))
    assert r.status is ParseStatus.OK
    assert r.ficha is not None and r.ficha.limitacao == "Uma instituição só."


def test_cercas_repaired():
    r = parse_completion("```json\n" + _json(_ficha()) + "\n```")
    assert r.status is ParseStatus.REPAIRED
    assert r.repairs == [REPAIR_FENCES]
    assert r.json_valid_first_try is False


def test_virgula_final_repaired():
    text = _json(_ficha()).replace('"pagina": 3', '"pagina": 3,')
    r = parse_completion(text)
    assert r.status is ParseStatus.REPAIRED
    assert REPAIR_TRAILING_COMMA in r.repairs
    assert r.json_valid_first_try is False


def test_cercas_mais_virgula_como_o_fake_ruim():
    text = "```json\n" + _json(_ficha()).replace('"pagina": 3', '"pagina": 3,') + "\n```"
    r = parse_completion(text)
    assert r.status is ParseStatus.REPAIRED
    assert r.repairs == [REPAIR_FENCES, REPAIR_TRAILING_COMMA]


def test_texto_antes_e_depois_repaired():
    r = parse_completion("Claro! Aqui está a ficha:\n" + _json(_ficha()) + "\nEspero ter ajudado.")
    assert r.status is ParseStatus.REPAIRED
    assert r.repairs == [REPAIR_EXTRACT_OBJECT]


def test_wrapper_cot_repaired_quando_nao_pedido():
    text = _json({"raciocinio": "Página 3 tem a métrica.", "ficha": _ficha()})
    r = parse_completion(text)
    assert r.status is ParseStatus.REPAIRED
    assert r.repairs == [REPAIR_UNWRAP]
    assert r.json_valid_first_try is False
    assert r.raciocinio == "Página 3 tem a métrica."
    assert r.ficha is not None


def test_wrapper_cot_ok_quando_pedido():
    text = _json({"raciocinio": "ok", "ficha": _ficha()})
    r = parse_completion(text, chain_of_thought=True)
    assert r.status is ParseStatus.OK
    assert r.json_valid_first_try is True
    assert r.raciocinio == "ok"


def test_cot_pedido_sem_envelope_repaired():
    r = parse_completion(_json(_ficha()), chain_of_thought=True)
    assert r.status is ParseStatus.REPAIRED
    assert r.repairs == [REPAIR_COT_NO_ENVELOPE]


@pytest.mark.parametrize("sentinela", ["N/A", "null", "None", "", "não declarada"])
def test_sentinela_repaired_e_vira_none(sentinela):
    r = parse_completion(_json(_ficha(limitacao=sentinela)))
    assert r.status is ParseStatus.REPAIRED
    assert r.repairs == [REPAIR_LIMITACAO_SENTINEL]
    assert r.ficha is not None and r.ficha.limitacao is None
    # JSON era válido e aceito pelo schema: conta como válido de primeira.
    assert r.json_valid_first_try is True


def test_limitacao_ausente_repaired():
    d = _ficha()
    del d["limitacao"]
    r = parse_completion(_json(d))
    assert r.status is ParseStatus.REPAIRED
    assert r.repairs == [REPAIR_LIMITACAO_MISSING]
    assert r.ficha is not None and r.ficha.limitacao is None


@pytest.mark.parametrize("pagina", ["3", "p. 3", "[p. 3]", "página 3", 3.0])
def test_pagina_string_para_int(pagina):
    r = parse_completion(_json(_ficha(evidencia={"trecho": TRECHO, "pagina": pagina})))
    assert r.status is ParseStatus.REPAIRED
    assert REPAIR_PAGINA_INT in r.repairs
    assert r.ficha is not None and r.ficha.evidencia.pagina == 3
    assert r.json_valid_first_try is False


def test_aspas_tipograficas():
    text = _json(_ficha()).replace('"', "“", 1)  # primeira aspa vira “
    r = parse_completion(text)
    assert r.status is ParseStatus.REPAIRED
    assert REPAIR_SMART_QUOTES in r.repairs


def test_campo_extra_failed_sem_reparo():
    r = parse_completion(_json(_ficha(confianca="alta")))
    assert r.status is ParseStatus.FAILED
    assert r.ficha is None
    assert r.json_valid_first_try is False
    assert r.error is not None and "confianca" in r.error


def test_envelope_com_chave_extra_nao_e_desembrulhado():
    r = parse_completion(_json({"raciocinio": "x", "ficha": _ficha(), "extra": 1}))
    assert r.status is ParseStatus.FAILED


@pytest.mark.parametrize(
    "lixo",
    [
        "Desculpe, não consigo ajudar com isso.",
        "",
        '{"problema": "truncado", "dados": "x", "metodo":',
        "[1, 2, 3]",
    ],
)
def test_lixo_failed(lixo):
    r = parse_completion(lixo)
    assert r.status is ParseStatus.FAILED
    assert r.ficha is None
    assert r.json_valid_first_try is False
    assert r.error


def test_schema_violado_error_legivel():
    r = parse_completion(_json(_ficha(evidencia={"trecho": "curto", "pagina": 0})))
    assert r.status is ParseStatus.FAILED
    assert r.error is not None
    assert r.error.startswith("Schema inválido:")
    assert "evidencia.trecho" in r.error and "evidencia.pagina" in r.error


def test_extract_first_object_ignora_chaves_em_strings():
    text = 'antes {"a": "tem } dentro", "b": {"c": 1}} depois {"x": 2}'
    assert extract_first_object(text) == '{"a": "tem } dentro", "b": {"c": 1}}'
    assert extract_first_object('{"a": 1') is None
    assert extract_first_object("sem objeto") is None


def test_remove_trailing_commas_preserva_strings():
    text = '{"a": "x, }", "b": [1, 2,], }'
    assert remove_trailing_commas(text) == '{"a": "x, }", "b": [1, 2] }'


def test_parser_nunca_altera_o_texto_de_entrada():
    text = "```json\n" + _json(_ficha()) + "\n```"
    copia = str(text)
    parse_completion(text)
    assert text == copia
