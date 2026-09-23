"""Linha de comando ``ficha``: comandos finos que só compõem o pacote.

Cada comando corresponde a uma etapa do notebook, para quem prefere rodar por partes::

    ficha ingest                       # PDFs de data/raw → cache de texto limpo
    ficha extract --strategy semantic --variant v_full --temperature 0 --repetition 1
    ficha audit --primary RUN --stability RUN --input-alt RUN --temp-alt RUN --prompt-alt RUN
    ficha cost [RUN ...]               # custo real (todas as execuções) × ingênuo
    ficha report --exemplo             # relatório de exemplo (≤ 3 páginas)
    ficha smoke                        # pipeline inteiro com FakeLLM em ~30 s

Nenhuma lógica de negócio mora aqui: seleção, prompt, parse, auditoria, custo e relatório
estão nos subpacotes. As chaves de API vêm só do ambiente (``Settings``), nunca de flags.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from ficha.audit import AuditSummary
from ficha.config import Settings, get_settings
from ficha.cost import CostComparison
from ficha.llm.fake import FakeBehavior
from ficha.types import Document, ExtractionRecord, GenerationParams

app = typer.Typer(
    name="ficha",
    help="Extração auditável de fichas comparativas (Atividade de Construção I).",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

REHEARSAL_BEHAVIOR = FakeBehavior(
    broken_json_rate=0.25,
    invent_trecho_rate=0.2,
    invent_limitacao_rate=0.4,
    unstable_rate=0.3,
)
"""Erros simulados no modo de ensaio — para a auditoria ter o que encontrar."""


# --------------------------------------------------------------------------- utilitários


def _settings(data_dir: Path | None = None, **overrides: Any) -> Settings:
    if data_dir is not None:
        overrides["data_dir"] = data_dir
    return get_settings(**overrides)


def _print_frame(frame: Any, title: str | None = None, max_rows: int = 30) -> None:
    """Imprime um ``DataFrame`` como tabela ``rich``."""
    table = Table(title=title, show_lines=False, header_style="bold")
    for col in frame.columns:
        table.add_column(str(col))
    for row in frame.head(max_rows).itertuples(index=False):
        table.add_row(*("" if v is None else str(v) for v in row))
    console.print(table)


def load_documents(raw_dir: Path, processed_dir: Path) -> list[Document]:
    """Carrega os PDFs de ``raw_dir`` usando o cache de texto limpo em ``processed_dir``."""
    from ficha.ingest import load_or_ingest

    pdfs = sorted(Path(raw_dir).glob("*.pdf"), key=lambda p: p.name)
    if not pdfs:
        raise typer.BadParameter(f"Nenhum PDF em {raw_dir}.")
    return [load_or_ingest(p, processed_dir) for p in pdfs]


# --------------------------------------------------------------------------- ingest


@app.command()
def ingest(
    raw_dir: Annotated[Path | None, typer.Option(help="Diretório dos PDFs.")] = None,
    processed_dir: Annotated[Path | None, typer.Option(help="Cache de texto limpo.")] = None,
) -> None:
    """Extrai e limpa o texto dos PDFs (cache em data/processed) e mostra o corpus."""
    from ficha.llm import approx_token_count
    from ficha.report.overview import corpus_frame

    s = _settings()
    docs = load_documents(raw_dir or s.raw_dir, processed_dir or s.processed_dir)
    frame = corpus_frame(docs, approx_token_count)
    _print_frame(frame, "Corpus (tokens ≈ caracteres/4, premissa declarada)", max_rows=100)


# --------------------------------------------------------------------------- extract


@app.command()
def extract(
    strategy: Annotated[str, typer.Option(help="first_pages | keyword | semantic")] = "semantic",
    variant: Annotated[str, typer.Option(help="Variante do prompt (ficha.prompts.VARIANTS).")] = (
        "v_full"
    ),
    temperature: Annotated[float | None, typer.Option(help="Padrão: Settings.")] = None,
    repetition: Annotated[int, typer.Option(help="1 = principal, 2 = estabilidade.")] = 1,
    backend: Annotated[str | None, typer.Option(help="fake | qwen_local | anthropic ...")] = None,
    limit: Annotated[int | None, typer.Option(help="Só os N primeiros artigos.")] = None,
    hashing_embedder: Annotated[
        bool, typer.Option(help="Embedder offline (sem download) para 'semantic'.")
    ] = False,
) -> None:
    """Roda uma execução (um run_id) e grava as saídas brutas em data/runs."""
    from ficha.extract.runner import ExtractionRunner
    from ficha.extract.store import RunStore
    from ficha.llm import build_client
    from ficha.prompts import build_prompt
    from ficha.select import HashingEmbedder, build_selector

    s = _settings()
    s.ensure_dirs()
    docs = load_documents(s.raw_dir, s.processed_dir)[:limit]
    client = build_client(s, backend=backend) if backend else build_client(s)
    selector = build_selector(strategy, s, HashingEmbedder() if hashing_embedder else None)
    params = GenerationParams(
        temperature=s.temperature if temperature is None else temperature,
        max_new_tokens=s.max_new_tokens,
        seed=s.seed,
    )
    runner = ExtractionRunner(client, selector, build_prompt(variant), params, RunStore(s.runs_dir))
    result = runner.run(docs, repetition=repetition)
    console.print(f"[bold]run_id:[/bold] {result.run_id}")
    console.print(f"status: {result.status_counts()}")


# --------------------------------------------------------------------------- audit


def audit_runs(
    runs: Mapping[str, Sequence[ExtractionRecord]],
    limitacao_gabarito: Mapping[str, bool] | None = None,
) -> AuditSummary:
    """Aplica as quatro verificações + comparação de prompts a um conjunto nomeado de execuções.

    ``runs`` precisa das chaves ``primary``, ``stability``, ``input_alt``, ``temp_alt`` e
    ``prompt_alt`` (os registros de cada execução).
    """
    from ficha.audit import build_audit_summary

    return build_audit_summary(
        runs["primary"],
        stability_rep=runs["stability"],
        alt_input=runs["input_alt"],
        t_alt=runs["temp_alt"],
        prompt_runs=(runs["primary"], runs["prompt_alt"]),
        limitacao_gabarito=limitacao_gabarito,
    )


def _load_runs(runs_dir: Path, ids: Mapping[str, str]) -> dict[str, list[ExtractionRecord]]:
    from ficha.extract.store import RunStore

    store = RunStore(runs_dir)
    return {k: store.load(v) for k, v in ids.items()}


RunOpt = Annotated[str, typer.Option(help="run_id em data/runs.")]


@app.command()
def audit(
    primary: RunOpt,
    stability: RunOpt,
    input_alt: RunOpt,
    temp_alt: RunOpt,
    prompt_alt: RunOpt,
) -> None:
    """As quatro verificações da Seção 4.4 + comparação de prompts (4.2)."""
    s = _settings()
    ids = {
        "primary": primary,
        "stability": stability,
        "input_alt": input_alt,
        "temp_alt": temp_alt,
        "prompt_alt": prompt_alt,
    }
    summary = audit_runs(_load_runs(s.runs_dir, ids))
    out = s.outputs_dir / "audit_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _print_audit(summary)
    console.print(f"Resumo completo em {out}")


def _print_audit(summary: AuditSummary) -> None:
    from ficha.report.assemble import (
        fidelity_block,
        input_block,
        prompt_conclusion,
        stability_block,
        temperature_block,
    )

    for block in (
        fidelity_block(summary),
        stability_block(summary),
        input_block(summary),
        temperature_block(summary),
    ):
        nums = (
            " · ".join(f"{k}: {v}" for k, v in block.numeros.items())
            if isinstance(block.numeros, Mapping)
            else ""
        )
        console.print(f"[bold]{block.titulo}[/bold] — {nums}\n  {block.conclusao}")
    if summary.prompt_comparison is not None:
        console.print(f"[bold]Prompts[/bold] — {prompt_conclusion(summary.prompt_comparison)}")
    console.print(f"[bold]Confiança[/bold] — {summary.confidence_distribution}")
    for f in summary.nao_defensaveis():
        console.print(f"  não defensável: {f.arquivo} — {'; '.join(f.motivos)}")


# --------------------------------------------------------------------------- cost


@app.command()
def cost(
    run_ids: Annotated[
        list[str] | None, typer.Argument(help="Execuções a somar (padrão: todas em data/runs).")
    ] = None,
) -> None:
    """Custo real (todas as execuções) × alternativa ingênua, com a premissa visível."""
    from ficha.cost import (
        PricePremise,
        compare_costs,
        cost_by_run,
        cost_of_records,
        cost_table,
        naive_cost_from_records,
    )
    from ficha.extract.store import RunStore
    from ficha.llm import approx_token_count

    s = _settings()
    store = RunStore(s.runs_dir)
    ids = run_ids or store.list_runs()
    if not ids:
        raise typer.BadParameter("Nenhuma execução em data/runs.")
    records = [r for rid in ids for r in store.load(rid)]
    docs = load_documents(s.raw_dir, s.processed_dir)
    premise = PricePremise.from_settings(s)
    real = cost_of_records(records, premise, label="real (todas as execuções)")
    naive = naive_cost_from_records(records, docs, approx_token_count, premise)
    comparison = compare_costs(real, naive)
    _print_frame(cost_table([*cost_by_run(records, premise), real, naive]), "Custo (Seção 4.5)")
    console.print(comparison.describe())


# --------------------------------------------------------------------------- report


@app.command()
def report(
    exemplo: Annotated[
        bool, typer.Option(help="Gera o relatório com conteúdo de exemplo (sem execuções).")
    ] = False,
    out: Annotated[Path | None, typer.Option(help="Caminho do PDF.")] = None,
) -> None:
    """Gera o relatório PDF (≤ 3 páginas). Sem --exemplo, use o notebook (célula 12)."""
    from ficha.report import build_report_pdf, count_pages, delivery_basename, sample_content

    if not exemplo:
        console.print(
            "O relatório real é montado no notebook (seção 12) a partir dos objetos da "
            "auditoria. Use --exemplo para ver o formato."
        )
        raise typer.Exit(code=1)
    content = sample_content()
    s = _settings()
    path = out or s.outputs_dir / f"{delivery_basename(content.integrantes)}_exemplo.pdf"
    build_report_pdf(content, path)
    console.print(f"PDF: {path} ({count_pages(path)} páginas)")


# --------------------------------------------------------------------------- smoke


@dataclass(slots=True)
class RehearsalResult:
    """Artefatos de um ensaio completo (ver :func:`run_rehearsal`)."""

    run_ids: dict[str, str]
    summary: AuditSummary
    cost: CostComparison
    table_csv: Path
    table_xlsx: Path
    pdf: Path
    pdf_pages: int
    seconds: float


def run_rehearsal(
    work_dir: Path,
    n_docs: int = 6,
    seed: int = 42,
    behavior: FakeBehavior | None = None,
    integrantes: Sequence[str] = ("<<INTEGRANTES: Nome Sobrenome>>",),
) -> RehearsalResult:
    """Pipeline inteiro com FakeLLM e PDFs sintéticos em ``work_dir`` (sem rede, sem GPU).

    Matriz de execuções (a mesma do notebook): (a) principal ``v_full × semantic × t0 × rep1``;
    (b) repetição idêntica; (c) ``v_sem_fewshot``; (d) ``first_pages``; (e) ``t_alt`` num
    subconjunto.
    """
    from ficha.cost import (
        PricePremise,
        compare_costs,
        cost_by_run,
        cost_of_records,
        naive_cost_from_records,
    )
    from ficha.extract.runner import ExtractionRunner
    from ficha.extract.store import RunStore
    from ficha.ingest.synthetic import MANIFEST_NAME, make_synthetic_corpus
    from ficha.llm import build_client
    from ficha.prompts import build_prompt
    from ficha.report import build_report_pdf, count_pages, delivery_basename, write_table
    from ficha.report.assemble import build_report_content
    from ficha.report.overview import selection_frame, strategy_summary
    from ficha.select import SELECTOR_NAMES, HashingEmbedder, build_selector

    t0 = time.perf_counter()
    s = _settings(data_dir=work_dir, model_backend="fake", seed=seed)
    s.ensure_dirs()
    make_synthetic_corpus(s.raw_dir, n=n_docs, seed=seed)
    manifest = json.loads((s.raw_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    gabarito = {a["arquivo"]: bool(a["has_limitations"]) for a in manifest["articles"]}
    docs = load_documents(s.raw_dir, s.processed_dir)

    client = build_client(s, behavior=behavior or REHEARSAL_BEHAVIOR)
    embedder = HashingEmbedder()
    store = RunStore(s.runs_dir)

    def run(
        strategy: str, variant: str, temp: float, rep: int, subset: Sequence[Document]
    ) -> tuple[str, list[ExtractionRecord]]:
        params = GenerationParams(temperature=temp, max_new_tokens=s.max_new_tokens, seed=seed)
        runner = ExtractionRunner(
            client, build_selector(strategy, s, embedder), build_prompt(variant), params, store
        )
        res = runner.run(subset, repetition=rep, show_progress=False)
        return res.run_id, res.records

    half = docs[: max(2, len(docs) // 2)]
    plan = {
        "primary": ("semantic", "v_full", s.temperature, 1, docs),
        "stability": ("semantic", "v_full", s.temperature, 2, docs),
        "prompt_alt": ("semantic", "v_sem_fewshot", s.temperature, 1, docs),
        "input_alt": ("first_pages", "v_full", s.temperature, 1, docs),
        "temp_alt": ("semantic", "v_full", s.temperature_alt, 1, half),
    }
    run_ids: dict[str, str] = {}
    runs: dict[str, list[ExtractionRecord]] = {}
    for key, args in plan.items():
        run_ids[key], runs[key] = run(*args)

    summary = audit_runs(runs, limitacao_gabarito=gabarito)
    all_records = [r for rs in runs.values() for r in rs]
    premise = PricePremise.from_settings(s)
    real = cost_of_records(all_records, premise, label="real (todas as execuções)")
    naive = naive_cost_from_records(all_records, docs, client.count_tokens, premise)
    comparison = compare_costs(real, naive)

    basename = delivery_basename(integrantes)
    fichas = [f.ficha for f in summary.fichas]
    extras = {f.arquivo: {"motivos_confianca": f.motivos} for f in summary.fichas}
    tables = write_table(fichas, s.outputs_dir, basename=basename, extras=extras)

    selectors = [build_selector(n, s, embedder) for n in SELECTOR_NAMES]
    strategies = strategy_summary(selection_frame(docs, selectors, client.count_tokens))
    content = build_report_content(
        integrantes=integrantes,
        modelo=f"{client.model_name} (modo de ensaio: FakeLLM, PDFs sintéticos)",
        summary=summary,
        cost=comparison,
        cost_per_run=cost_by_run(all_records, premise),
        strategy_table=strategies[["caracteres_medios", "tokens_medios", "tokens_corpus"]],
        leak_runs={"principal": runs["primary"], "v_sem_fewshot": runs["prompt_alt"]},
    )
    pdf = build_report_pdf(content, s.outputs_dir / f"{basename}.pdf")
    return RehearsalResult(
        run_ids=run_ids,
        summary=summary,
        cost=comparison,
        table_csv=tables.csv,
        table_xlsx=tables.xlsx,
        pdf=pdf,
        pdf_pages=count_pages(pdf),
        seconds=time.perf_counter() - t0,
    )


@app.command()
def smoke(
    n_docs: Annotated[int, typer.Option(help="Artigos sintéticos.")] = 6,
    keep: Annotated[
        Path | None, typer.Option(help="Diretório para manter os artefatos (padrão: temporário).")
    ] = None,
) -> None:
    """Demonstração de 30 s: pipeline inteiro com FakeLLM + PDFs sintéticos."""
    with tempfile.TemporaryDirectory(prefix="ficha_smoke_") as tmp:
        work = keep or Path(tmp)
        res = run_rehearsal(work, n_docs=n_docs)
        console.rule("[bold]ficha smoke — modo de ensaio (FakeLLM)[/bold]")
        for key, rid in res.run_ids.items():
            console.print(f"  {key:<10} {rid}")
        _print_audit(res.summary)
        console.print(f"[bold]Custo[/bold] — {res.cost.describe()}")
        console.print(f"Tabela: {res.table_csv.name}, {res.table_xlsx.name}")
        console.print(f"Relatório: {res.pdf.name} ({res.pdf_pages} páginas)")
        console.print(f"Concluído em {res.seconds:.1f} s em {work}")


def main() -> None:  # pragma: no cover - ponto de entrada
    """Ponto de entrada para ``python -m ficha.cli``."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
