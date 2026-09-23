"""Orquestração da extração: seleção → prompt → modelo → parse → registro persistido.

Para cada artigo: ``selector.select`` → ``builder.render`` → ``client.generate`` →
``parse_completion`` → :class:`ExtractionRecord` → ``store.append`` (gravado na hora: se a sessão
do Colab cair no artigo 15, os 14 primeiros estão salvos).

Robustez: exceção do cliente (timeout, falta de memória, erro da API) **não** derruba a
execução — vira um registro ``FAILED`` com ``error``. JSON malformado também não: é o parser
que decide entre ``OK``/``REPAIRED``/``FAILED`` (Seção 4.2: "seu código detecta e trata;
não quebra").
"""

from __future__ import annotations

import random
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np

from ficha.extract.parser import ParseResult, parse_completion
from ficha.extract.store import RunStore
from ficha.prompts.builder import PromptBuilder
from ficha.prompts.variants import VARIANTS
from ficha.types import (
    ContextSelector,
    Document,
    ExtractionRecord,
    GenerationParams,
    LLMClient,
    ParseStatus,
    Usage,
)


def set_seeds(seed: int | None) -> None:
    """Fixa as sementes de ``random``, ``numpy`` e — se já importado — ``torch`` (Seção 7)."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def reparse(record: ExtractionRecord) -> ParseResult:
    """Refaz o parse da saída bruta de um registro (determinístico).

    O registro guarda ``raw_output`` intacto, então os reparos podem ser recomputados a qualquer
    momento — por exemplo, para registros gravados antes de ``ExtractionRecord.repairs`` existir,
    ou para conferir que ``record.repairs`` bate com o parser atual.
    """
    features = VARIANTS.get(record.prompt_variant)
    cot = features.chain_of_thought if features is not None else False
    return parse_completion(record.raw_output, chain_of_thought=cot)


@dataclass(slots=True)
class RunResult:
    """Resultado de uma execução: o id, os registros (na ordem dos documentos) e o manifest."""

    run_id: str
    records: list[ExtractionRecord]
    manifest: dict[str, Any] = field(default_factory=dict)

    def status_counts(self) -> dict[str, int]:
        """Quantos registros em cada ``ParseStatus``."""
        counts = Counter(r.parse_status.value for r in self.records)
        return {s.value: counts.get(s.value, 0) for s in ParseStatus}


def _summarize(records: Sequence[ExtractionRecord]) -> dict[str, Any]:
    usage = sum((r.usage for r in records), Usage(0, 0))
    n = len(records)
    counts = Counter(r.parse_status.value for r in records)
    first_try = sum(r.json_valid_first_try for r in records)
    with_ficha = [r for r in records if r.ficha is not None]
    n_null = sum(r.ficha.limitacao is None for r in with_ficha if r.ficha is not None)
    return {
        "n_registros": n,
        "status": {s.value: counts.get(s.value, 0) for s in ParseStatus},
        "json_valido_de_primeira": first_try,
        "taxa_json_valido_de_primeira": round(first_try / n, 4) if n else None,
        "limitacao_null": n_null,
        "reparos": dict(Counter(rep for r in records for rep in r.repairs).most_common()),
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total,
        },
        "latencia_total_s": round(sum(r.latency_s for r in records), 3),
    }


class ExtractionRunner:
    """Executa a extração de um conjunto de artigos com uma configuração fixa e declarada."""

    def __init__(
        self,
        client: LLMClient,
        selector: ContextSelector,
        builder: PromptBuilder,
        params: GenerationParams,
        store: RunStore,
    ) -> None:
        self.client = client
        self.selector = selector
        self.builder = builder
        self.params = params
        self.store = store

    # ------------------------------------------------------------------ manifest
    def _base_manifest(
        self, run_id: str, docs: Sequence[Document], repetition: int, started_at: str
    ) -> dict[str, Any]:
        from ficha import __version__

        return {
            "run_id": run_id,
            "ficha_version": __version__,
            "modelo": self.client.model_name,
            "cliente": type(self.client).__name__,
            "prompt": {
                "variante": self.builder.variant,
                "features": asdict(self.builder.features),
                "descricao": self.builder.features.describe(),
                "template_sha": self.builder.template_sha(),
                "blocos": self.builder.markers_present(),
            },
            "estrategia": {"nome": self.selector.name, "params": {}},
            # Hardware e precisão (Seção 4.3): declarar onde rodou, não só qual modelo.
            "dispositivo": getattr(self.client, "device_info", None),
            "geracao": asdict(self.params),
            "temperatura": self.params.temperature,
            "seed": self.params.seed,
            "repeticao": repetition,
            "n_docs": len(docs),
            "arquivos": [d.arquivo for d in docs],
            "iniciado_em": started_at,
            "concluido_em": None,
            "status_execucao": "em_andamento",
        }

    # ------------------------------------------------------------------ execução
    def _extract_one(
        self, doc: Document, run_id: str, repetition: int
    ) -> tuple[ExtractionRecord, dict[str, Any]]:
        """Um artigo → um registro (nunca levanta por erro do modelo) + params da estratégia."""
        context = self.selector.select(doc)
        context_text = context.render()
        prompt = self.builder.render(context)
        base: dict[str, Any] = {
            "run_id": run_id,
            "arquivo": doc.arquivo,
            "strategy": context.strategy,
            "prompt_variant": prompt.variant,
            "prompt_sha": prompt.sha256(),
            "temperature": self.params.temperature,
            "seed": self.params.seed,
            "repetition": repetition,
            "context_text": context_text,
            "context_pages": context.pages,
            "doc_sha256": doc.sha256(),
        }
        context_params = dict(context.params)

        t0 = time.perf_counter()
        try:
            completion = self.client.generate(prompt.system, prompt.user, self.params)
        except Exception as exc:
            failed = ExtractionRecord(
                **base,
                model=self.client.model_name,
                raw_output="",
                usage=Usage(0, 0),
                latency_s=time.perf_counter() - t0,
                json_valid_first_try=False,
                parse_status=ParseStatus.FAILED,
                ficha=None,
                error=f"Erro na chamada ao modelo: {type(exc).__name__}: {exc}",
            )
            return failed, context_params

        parsed = parse_completion(
            completion.text, chain_of_thought=self.builder.features.chain_of_thought
        )
        # ``error`` só é preenchido quando não há ficha; os reparos vão em ``repairs`` (e podem
        # ser recomputados com ``reparse(record)`` a partir de ``raw_output``).
        error = parsed.error
        if parsed.ficha is None and completion.raw.get("finish_reason") == "length":
            error = f"{error} | saída truncada no limite de max_new_tokens"
        record = ExtractionRecord(
            **base,
            model=completion.model,
            raw_output=completion.text,
            usage=completion.usage,
            latency_s=completion.latency_s,
            json_valid_first_try=parsed.json_valid_first_try,
            parse_status=parsed.status,
            ficha=parsed.ficha,
            error=error,
            repairs=list(parsed.repairs),
        )
        return record, context_params

    def run(
        self,
        docs: Sequence[Document],
        repetition: int = 1,
        run_id: str | None = None,
        show_progress: bool = True,
    ) -> RunResult:
        """Extrai uma ficha por documento e persiste tudo em ``store``.

        ``run_id=None`` gera um id novo (formato do ARCHITECTURE.md). As sementes são fixadas
        no início com ``params.seed``.
        """
        set_seeds(self.params.seed)
        started_at = datetime.now(UTC).isoformat()
        if run_id is None:
            run_id = self.store.new_run_id(
                self.selector.name, self.builder.variant, self.params.temperature, repetition
            )
        manifest = self._base_manifest(run_id, docs, repetition, started_at)
        self.store.write_manifest(run_id, manifest)

        strategy_params: dict[str, Any] = {}
        records: list[ExtractionRecord] = []
        iterable: Any = docs
        if show_progress:
            from tqdm.auto import tqdm

            iterable = tqdm(docs, desc=f"{self.builder.variant} t={self.params.temperature}")
        for doc in iterable:
            record, strategy_params = self._extract_one(doc, run_id, repetition)
            self.store.append(run_id, record)
            records.append(record)

        manifest["estrategia"]["params"] = strategy_params
        manifest["modelo_reportado"] = sorted({r.model for r in records})
        manifest["resumo"] = _summarize(records)
        manifest["concluido_em"] = datetime.now(UTC).isoformat()
        manifest["status_execucao"] = "concluida"
        self.store.write_manifest(run_id, manifest)
        return RunResult(run_id=run_id, records=records, manifest=manifest)
