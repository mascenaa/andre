from __future__ import annotations

import re
from datetime import datetime

import pytest

from ficha.extract.store import RunStore, format_temperature
from ficha.schema import Evidencia, FichaExtraida
from ficha.types import ExtractionRecord, ParseStatus, Usage

RUN_ID_RE = re.compile(r"^\d{8}-\d{6}_[\w.\-]+_[\w.\-]+_t\d+\.\d+_rep\d+(_\d+)?$")


def _record(run_id: str, arquivo: str = "a.pdf", ficha: bool = True) -> ExtractionRecord:
    return ExtractionRecord(
        run_id=run_id,
        arquivo=arquivo,
        strategy="first_pages",
        prompt_variant="v_full",
        prompt_sha="abc123abc123",
        model="fake",
        temperature=0.0,
        seed=42,
        repetition=1,
        context_text="[p. 1]\ntexto",
        context_pages=(1, 2),
        raw_output='{"x": 1}',
        usage=Usage(10, 5),
        latency_s=0.1,
        json_valid_first_try=ficha,
        parse_status=ParseStatus.OK if ficha else ParseStatus.FAILED,
        ficha=FichaExtraida(
            problema="p",
            dados="d",
            metodo="m",
            metrica="x",
            limitacao=None,
            evidencia=Evidencia(trecho="um trecho com mais de vinte chars", pagina=1),
        )
        if ficha
        else None,
        error=None if ficha else "JSON inválido",
    )


def test_format_temperature():
    assert format_temperature(0.0) == "0.0"
    assert format_temperature(0.7) == "0.7"
    assert format_temperature(0.25) == "0.25"
    assert format_temperature(1) == "1.0"


def test_new_run_id_formato_e_sem_colisao(tmp_path):
    store = RunStore(tmp_path)
    now = datetime(2026, 9, 23, 10, 11, 12)
    a = store.new_run_id("first_pages", "v_full", 0.0, 1, now=now)
    b = store.new_run_id("first_pages", "v_full", 0.0, 1, now=now)
    assert a == "20260923-101112_first_pages_v_full_t0.0_rep1"
    assert b == a + "_2"
    assert RUN_ID_RE.match(a) and RUN_ID_RE.match(b)
    assert store.run_dir(a).is_dir()


def test_run_id_saneia_caracteres(tmp_path):
    run_id = RunStore(tmp_path).new_run_id("semantic k=6", "v/x", 0.7, 2)
    assert "/" not in run_id and " " not in run_id
    assert run_id.endswith("_t0.7_rep2")


def test_append_load_roundtrip(tmp_path):
    store = RunStore(tmp_path)
    run_id = store.new_run_id("first_pages", "v_full", 0.0)
    recs = [_record(run_id, "a.pdf"), _record(run_id, "b.pdf", ficha=False)]
    for r in recs:
        store.append(run_id, r)
    loaded = store.load(run_id)
    assert [r.to_dict() for r in loaded] == [r.to_dict() for r in recs]
    assert loaded[0].context_pages == (1, 2)
    assert loaded[1].ficha is None


def test_manifest_e_list_runs(tmp_path):
    store = RunStore(tmp_path / "runs")
    assert store.list_runs() == []
    r1 = store.new_run_id("s", "v", 0.0, now=datetime(2026, 1, 1))
    r2 = store.new_run_id("s", "v", 0.7, now=datetime(2026, 1, 2))
    store.write_manifest(r2, {"modelo": "fake", "temperatura": 0.7})
    store.append(r1, _record(r1))
    assert store.list_runs() == [r1, r2]
    assert store.manifest(r2)["temperatura"] == 0.7
    with pytest.raises(FileNotFoundError):
        store.manifest(r1)
    with pytest.raises(FileNotFoundError):
        store.load(r2)
