from __future__ import annotations

from dataclasses import dataclass

from ficha.extract.parser import parse_completion
from ficha.extract.runner import ExtractionRunner, reparse
from ficha.extract.store import RunStore
from ficha.prompts import build_prompt
from ficha.types import (
    Chunk,
    Completion,
    Context,
    Document,
    ExtractionRecord,
    GenerationParams,
    Page,
    ParseStatus,
)


@dataclass
class FirstPagesStub:
    """Seletor trivial (não depende do implementer-1): as N primeiras páginas inteiras."""

    n: int = 3

    @property
    def name(self) -> str:
        return "first_pages"

    def select(self, doc: Document) -> Context:
        chunks = tuple(
            Chunk(arquivo=doc.arquivo, page=p.number, text=p.text, start=0, end=len(p.text))
            for p in doc.pages[: self.n]
        )
        return Context(doc.arquivo, self.name, chunks, params={"n_pages": self.n})


class ExplodingClient:
    model_name = "explode/v1"

    def generate(self, system, user, params) -> Completion:
        raise TimeoutError("GPU sumiu")

    def count_tokens(self, text: str) -> int:
        return 1


def _runner(client, tmp_path, params, variant="v_full"):
    return ExtractionRunner(
        client=client,
        selector=FirstPagesStub(),
        builder=build_prompt(variant),
        params=params,
        store=RunStore(tmp_path),
    )


def test_run_gera_persiste_e_recarrega(fake_llm, docs, params, tmp_path):
    runner = _runner(fake_llm, tmp_path, params)
    result = runner.run(docs, show_progress=False)

    assert len(result.records) == len(docs)
    assert result.run_id.endswith("_first_pages_v_full_t0.0_rep1")
    for rec, doc in zip(result.records, docs, strict=True):
        assert rec.arquivo == doc.arquivo
        assert rec.parse_status is ParseStatus.OK
        assert rec.json_valid_first_try
        assert rec.ficha is not None
        ctx = FirstPagesStub().select(doc)
        assert rec.context_text == ctx.render()
        assert rec.context_pages == (1, 2, 3)
        assert rec.doc_sha256 == doc.sha256()
        assert rec.prompt_sha == build_prompt("v_full").render(ctx).sha256()
        assert rec.ficha.evidencia.trecho in rec.context_text
        assert rec.usage.input_tokens > 0 and rec.usage.output_tokens > 0
        assert rec.raw_output  # saída bruta guardada
        assert rec.error is None

    loaded = RunStore(tmp_path).load(result.run_id)
    assert [r.to_dict() for r in loaded] == [r.to_dict() for r in result.records]
    assert [ExtractionRecord.from_dict(r.to_dict()).to_dict() for r in loaded] == [
        r.to_dict() for r in loaded
    ]


def test_abstencao_no_doc_sem_limitacao(fake_llm, doc_sem_limitacao, params, tmp_path):
    result = _runner(fake_llm, tmp_path, params).run([doc_sem_limitacao], show_progress=False)
    assert result.records[0].ficha is not None
    assert result.records[0].ficha.limitacao is None
    assert result.manifest["resumo"]["limitacao_null"] == 1


def _varied_docs(n: int) -> list[Document]:
    """Documentos com textos distintos (com t=0 o fake é determinístico por prompt)."""
    from conftest import make_document

    out = []
    for i in range(n):
        base = make_document(f"artigo_{i:02d}.pdf", with_limitations=i % 2 == 0)
        first = Page(1, f"Study number {i} of the benchmark collection. " + base.pages[0].text)
        out.append(Document(base.arquivo, (first, *base.pages[1:])))
    return out


def test_manifest_declara_tudo(fake_llm, docs, params, tmp_path):
    result = _runner(fake_llm, tmp_path, params, "v_sem_fewshot").run(
        docs, repetition=2, show_progress=False
    )
    m = RunStore(tmp_path).manifest(result.run_id)
    assert m == result.manifest
    assert m["modelo"] == fake_llm.model_name
    assert m["prompt"]["variante"] == "v_sem_fewshot"
    assert m["prompt"]["features"]["few_shot"] is False
    assert "### EXEMPLOS" not in m["prompt"]["blocos"]
    assert m["estrategia"] == {"nome": "first_pages", "params": {"n_pages": 3}}
    assert m["temperatura"] == 0.0 and m["seed"] == 42
    assert m["repeticao"] == 2
    assert m["n_docs"] == 2
    assert m["status_execucao"] == "concluida"
    assert m["iniciado_em"] and m["concluido_em"]
    total_in = sum(r.usage.input_tokens for r in result.records)
    assert m["resumo"]["usage"]["input_tokens"] == total_in
    assert m["resumo"]["status"] == {"ok": 2, "repaired": 0, "failed": 0}
    assert m["resumo"]["taxa_json_valido_de_primeira"] == 1.0


def test_temperatura_zero_e_deterministica(fake_llm, docs, params, tmp_path):
    runner = _runner(fake_llm, tmp_path, params)
    a = runner.run(docs, repetition=1, show_progress=False)
    b = runner.run(docs, repetition=2, show_progress=False)
    assert [r.raw_output for r in a.records] == [r.raw_output for r in b.records]
    assert a.run_id != b.run_id


def test_fake_ruim_mistura_status_sem_levantar(fake_llm_ruim, params, tmp_path):
    many = _varied_docs(30)
    result = _runner(fake_llm_ruim, tmp_path, params).run(many, show_progress=False)
    counts = result.status_counts()
    assert counts["ok"] > 0
    assert counts["repaired"] > 0
    assert sum(counts.values()) == 30
    for rec in result.records:
        if rec.parse_status is ParseStatus.REPAIRED:
            assert not rec.json_valid_first_try
            assert rec.ficha is not None
            assert rec.repairs
            assert reparse(rec).repairs == rec.repairs  # recomputável da saída bruta
        if rec.parse_status is ParseStatus.OK:
            assert rec.repairs == []
    assert len(RunStore(tmp_path).load(result.run_id)) == 30


def test_repairs_persistidos_round_trip(params, tmp_path):
    from ficha.llm.fake import FakeBehavior, FakeLLM

    fake = FakeLLM(behavior=FakeBehavior(broken_json_rate=1.0))
    result = _runner(fake, tmp_path, params).run(_varied_docs(3), show_progress=False)
    for rec in result.records:
        assert rec.parse_status is ParseStatus.REPAIRED
        assert rec.repairs == ["cercas_de_codigo", "virgula_final"]
        assert rec.error is None
    loaded = RunStore(tmp_path).load(result.run_id)
    assert [r.repairs for r in loaded] == [r.repairs for r in result.records]
    assert [r.to_dict() for r in loaded] == [r.to_dict() for r in result.records]
    assert result.manifest["resumo"]["reparos"] == {"cercas_de_codigo": 3, "virgula_final": 3}


def test_registro_antigo_sem_repairs_carrega(fake_llm, docs, params, tmp_path):
    import json

    store = RunStore(tmp_path)
    result = _runner(fake_llm, tmp_path, params).run(docs, show_progress=False)
    path = store.records_path(result.run_id)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for d in lines:
        del d["repairs"]
    path.write_text("".join(json.dumps(d) + "\n" for d in lines), encoding="utf-8")
    loaded = store.load(result.run_id)
    assert all(r.repairs == [] for r in loaded)
    assert all(reparse(r).repairs == [] for r in loaded)


def test_truncamento_do_fake_vira_failed_com_nota(params, tmp_path):
    from ficha.llm.fake import FakeBehavior, FakeLLM

    fake = FakeLLM(behavior=FakeBehavior(truncate_rate=1.0))
    result = _runner(fake, tmp_path, params).run(_varied_docs(4), show_progress=False)
    for rec in result.records:
        assert rec.parse_status is ParseStatus.FAILED
        assert rec.ficha is None
        assert rec.raw_output  # a saída cortada é guardada intacta
        assert "saída truncada" in rec.error
    assert result.manifest["resumo"]["status"]["failed"] == 4


def test_parser_falha_vira_failed(docs, params, tmp_path):
    class LixoClient:
        model_name = "lixo/v1"

        def generate(self, system, user, params) -> Completion:
            from ficha.types import Usage

            return Completion("não sei", Usage(10, 2), self.model_name, 0.01, params)

        def count_tokens(self, text):
            return 1

    result = _runner(LixoClient(), tmp_path, params).run(docs, show_progress=False)
    assert all(r.parse_status is ParseStatus.FAILED for r in result.records)
    assert all(r.raw_output == "não sei" for r in result.records)
    assert all(r.error for r in result.records)


def test_excecao_do_cliente_vira_failed(docs, params, tmp_path):
    result = _runner(ExplodingClient(), tmp_path, params).run(docs, show_progress=False)
    assert len(result.records) == 2
    for rec in result.records:
        assert rec.parse_status is ParseStatus.FAILED
        assert rec.ficha is None
        assert "TimeoutError" in rec.error and "GPU sumiu" in rec.error
        assert rec.raw_output == ""
        assert rec.model == "explode/v1"
    assert result.manifest["resumo"]["status"]["failed"] == 2


def test_cot_usa_parse_com_envelope(fake_llm, docs, tmp_path):
    params = GenerationParams(temperature=0.0, seed=1)
    result = _runner(fake_llm, tmp_path, params, "v_cot").run(docs, show_progress=False)
    # O fake não usa o envelope: com CoT pedido, isso é um reparo registrado.
    for rec in result.records:
        assert rec.parse_status is ParseStatus.REPAIRED
        assert reparse(rec).repairs == ["cot_sem_envelope"]
        assert parse_completion(rec.raw_output).status is ParseStatus.OK


def test_run_id_explicito(fake_llm, docs, params, tmp_path):
    result = _runner(fake_llm, tmp_path, params).run(docs, run_id="meu_run", show_progress=False)
    assert result.run_id == "meu_run"
    assert all(r.run_id == "meu_run" for r in result.records)
    assert "meu_run" in RunStore(tmp_path).list_runs()
