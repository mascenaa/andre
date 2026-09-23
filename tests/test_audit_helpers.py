"""Construtores de ExtractionRecord para os testes de auditoria e custo.

Não depende de ``ficha.extract``: reproduz o mínimo que o runner faria (prompt com
delimitadores → FakeLLM → json.loads → validação), para gerar registros realistas.
"""

from __future__ import annotations

import json
import re
from typing import Any

from conftest import make_context, make_document
from ficha.llm.fake import FakeBehavior, FakeLLM
from ficha.schema import FichaExtraida
from ficha.types import Context, Document, ExtractionRecord, GenerationParams, ParseStatus, Usage

CONTEXT = make_context(make_document()).render()

TRECHO_OK = (
    "Existing approaches rely on hand-crafted risk scores that generalize poorly across "
    "institutions."
)


def ficha_dict(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "problema": "Prever readmissão hospitalar em 30 dias.",
        "dados": "MIMIC-IV, 2008-2019, 431.231 internações.",
        "metodo": "LightGBM com atributos estruturados e TF-IDF das notas.",
        "metrica": "AUROC e AUPRC.",
        "limitacao": "Dados de uma única instituição.",
        "evidencia": {"trecho": TRECHO_OK, "pagina": 1},
    }
    evid = overrides.pop("evidencia", None)
    base.update(overrides)
    if evid is not None:
        base["evidencia"] = {**base["evidencia"], **evid}
    return base


def make_record(
    arquivo: str = "artigo_01.pdf",
    ficha: dict[str, Any] | FichaExtraida | None = None,
    *,
    failed: bool = False,
    context_text: str = CONTEXT,
    run_id: str = "run_a",
    strategy: str = "first_pages",
    variant: str = "v_full",
    temperature: float = 0.0,
    seed: int | None = 42,
    repetition: int = 1,
    parse_status: ParseStatus | None = None,
    json_valid_first_try: bool | None = None,
    usage: Usage | None = None,
) -> ExtractionRecord:
    if failed:
        fx = None
    elif isinstance(ficha, FichaExtraida):
        fx = ficha
    else:
        fx = FichaExtraida.model_validate(ficha if ficha is not None else ficha_dict())
    status = parse_status or (ParseStatus.FAILED if fx is None else ParseStatus.OK)
    return ExtractionRecord(
        run_id=run_id,
        arquivo=arquivo,
        strategy=strategy,
        prompt_variant=variant,
        prompt_sha="abc123",
        model="fake",
        temperature=temperature,
        seed=seed,
        repetition=repetition,
        context_text=context_text,
        context_pages=(1, 2, 3),
        raw_output="{}",
        usage=usage or Usage(1000, 100),
        latency_s=0.01,
        json_valid_first_try=(
            json_valid_first_try if json_valid_first_try is not None else status == ParseStatus.OK
        ),
        parse_status=status,
        ficha=fx,
    )


def _parse(text: str) -> tuple[FichaExtraida | None, ParseStatus, bool]:
    """Parser mínimo de teste: OK de primeira, reparo de cerca + vírgula, ou falha."""
    try:
        return FichaExtraida.model_validate(json.loads(text)), ParseStatus.OK, True
    except ValueError:
        pass
    fixed = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    try:
        return FichaExtraida.model_validate(json.loads(fixed)), ParseStatus.REPAIRED, False
    except ValueError:
        return None, ParseStatus.FAILED, False


def record_from_fake(
    doc: Document,
    behavior: FakeBehavior | None = None,
    *,
    params: GenerationParams | None = None,
    context: Context | None = None,
    run_id: str = "run_fake",
    variant: str = "v_full",
    repetition: int = 1,
) -> ExtractionRecord:
    """Simula o runner: contexto → prompt delimitado → FakeLLM → parse."""
    params = params or GenerationParams()
    ctx = context or make_context(doc)
    llm = FakeLLM(behavior=behavior or FakeBehavior())
    user = f"Extraia a ficha.\n<<<ARTIGO>>>\n{ctx.render()}\n<<<FIM_ARTIGO>>>"
    comp = llm.generate("Você é um assistente de pesquisa.", user, params)
    fx, status, first = _parse(comp.text)
    return ExtractionRecord(
        run_id=run_id,
        arquivo=doc.arquivo,
        strategy=ctx.strategy,
        prompt_variant=variant,
        prompt_sha="fake",
        model=comp.model,
        temperature=params.temperature,
        seed=params.seed,
        repetition=repetition,
        context_text=ctx.render(),
        context_pages=ctx.pages,
        raw_output=comp.text,
        usage=comp.usage,
        latency_s=comp.latency_s,
        json_valid_first_try=first,
        parse_status=status,
        ficha=fx,
    )


def test_helpers_produzem_registros_validos() -> None:
    r = make_record()
    assert r.ficha is not None and r.parse_status == ParseStatus.OK
    rf = record_from_fake(make_document())
    assert rf.parse_status == ParseStatus.OK and rf.ficha is not None
    rb = record_from_fake(make_document(), FakeBehavior(broken_json_rate=1.0))
    assert rb.parse_status == ParseStatus.REPAIRED and not rb.json_valid_first_try
