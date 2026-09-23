"""Monta o :class:`~ficha.report.content.ReportContent` a partir dos objetos da auditoria.

É aqui — e não no notebook — que números viram frases e tabelas compactas. Assim o notebook
só orquestra (ADR 0001) e **nenhum número do relatório é digitado à mão**: tudo sai de
:class:`~ficha.audit.AuditSummary`, :class:`~ficha.audit.PromptComparison` e
:class:`~ficha.cost.CostComparison`.

Os textos de justificativa (por que esta estratégia, este modelo, esta temperatura) vêm de
:class:`Narrative`, cujos padrões resumem os ADRs; o grupo revisa e sobrescreve o que quiser.
As conclusões de cada bloco da auditoria são geradas a partir dos números, mas também podem
ser sobrescritas em ``Narrative.conclusoes`` depois que o grupo ler os resultados.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ficha.audit import AuditSummary, PromptComparison
from ficha.cost import CostComparison, CostReport
from ficha.prompts import VARIANTS, describe_diff
from ficha.report.content import (
    AuditBlock,
    AuditResults,
    CostSection,
    PromptVersion,
    ReportContent,
)

_FEATURE_LABELS = {
    "role": "papel",
    "delimiters": "delimitadores",
    "few_shot": "few-shot (com caso null)",
    "abstention": "abstenção",
    "chain_of_thought": "cadeia de raciocínio",
}

_PROMPT_ROWS: dict[str, str] = {
    "rate_json_valid_first_try": "JSON válido de primeira",
    "rate_parse_failed": "parse falhou (sem ficha)",
    "rate_null_when_expected": "null correto quando não há limitação",
    "rate_fidelity": "trecho encontrado (fidelidade)",
    "rate_page_ok": "página correta",
    "rate_identical_entre_variantes": "fichas idênticas entre versões",
    "input_tokens": "tokens de entrada",
}


def pct(value: float | None, digits: int = 0) -> str:
    """Taxa 0–1 como porcentagem pt-BR (``0.955`` → ``"96%"``); ``None`` vira ``"—"``."""
    if value is None or pd.isna(value):
        return "—"
    return f"{value * 100:.{digits}f}%".replace(".", ",")


def temp(value: float | None) -> str:
    """Temperatura em pt-BR (``0.7`` → ``"0,7"``)."""
    return "—" if value is None else f"{value:.1f}".replace(".", ",")


def features_label(variant: str) -> str:
    """Técnicas ligadas numa variante, em português (``"papel, delimitadores, ..."``)."""
    feats = VARIANTS[variant]
    on = [label for key, label in _FEATURE_LABELS.items() if getattr(feats, key)]
    return ", ".join(on)


@dataclass(slots=True)
class Narrative:
    """Textos de justificativa do relatório. Padrões resumem os ADRs 0002–0005."""

    estrategia_escolhida: str = (
        "semantic — trechos selecionados por similaridade dentro do próprio artigo "
        "(keyword como alternativa)"
    )
    estrategia_justificativa: str = (
        "Cada artigo é segmentado em blocos de 1.200 caracteres (~300 tokens, um parágrafo "
        "típico) com sobreposição de 200 (uma frase inteira, para não cortar a evidência); 8 "
        "consultas em português, uma ou duas por campo da ficha, recuperam os blocos mais "
        "próximos com um embedder multilíngue, reordenados por página. As primeiras páginas "
        "quase nunca trazem a limitação declarada e a busca por palavra-chave depende do nome "
        "da seção. A escolha é testada na verificação 4.4c (ADR 0002)."
    )
    modelo_justificativa: str = (
        "Qwen2.5-3B-Instruct cabe na T4 gratuita em meia precisão (~6 GB + contexto) e é o "
        "equilíbrio recomendado: o 1,5B erra demais nesta tarefa e o 7B exige 4 bits. Erros do "
        "modelo pequeno são matéria-prima da auditoria, não impedimento."
    )
    temperatura_justificativa: str = (
        "Extração é tarefa de fidelidade, não de criatividade: queremos a saída mais provável "
        "e reprodutível entre execuções (seed fixa). A verificação 4.4d mede o que muda com "
        "uma temperatura mais alta."
    )
    o_que_faria_diferente: list[str] = field(
        default_factory=lambda: [
            "Anotar à mão um gabarito de alguns artigos para medir acurácia, não só "
            "consistência e fidelidade.",
            "Comparar o modelo local com um modelo por API nas mesmas 19 fichas.",
            "Testar a variante com raciocínio em campo separado (v_cot) e medir se compensa.",
        ]
    )
    declaracao_uso_ia: str = (
        "Usamos assistentes de IA para escrever partes do código e revisar textos. As decisões "
        "de projeto, os números e as conclusões foram verificados pelo grupo, que sabe "
        "explicar cada linha do código entregue."
    )
    titulo: str = "A ficha que você teria de defender — extração auditável de 19 artigos"
    conclusoes: dict[str, str] = field(default_factory=dict)
    """Sobrescritas das conclusões automáticas: chaves ``fidelidade``, ``estabilidade``,
    ``entrada``, ``temperatura``, ``prompt``, ``custo``."""


# --------------------------------------------------------------------------- blocos


def _top_fields(rates: Mapping[str, float], k: int = 3) -> str:
    changed = sorted(((f, r) for f, r in rates.items() if r > 0), key=lambda t: -t[1])[:k]
    return ", ".join(f"{f} {pct(r)}" for f, r in changed) or "nenhum"


def fidelity_block(summary: AuditSummary) -> AuditBlock:
    """4.4a — quantos trechos existem no texto enviado.

    ``failures()`` mistura dois casos que o relatório separa: trecho devolvido mas ausente do
    texto enviado (invenção) e extração sem ficha (``method == "none"``, nada a verificar).
    """
    fid = summary.fidelity
    inventados = [r.arquivo for r in fid.failures() if r.method != "none"]
    sem_ficha = [r.arquivo for r in fid.failures() if r.method == "none"]
    by = fid.by_method
    numeros = {
        "trecho encontrado": f"{fid.n_found}/{fid.n} ({pct(fid.rate_found)})",
        "página correta": f"{fid.n_page_ok}/{fid.n}",
        "exato / aproximado / sem ficha": (
            f"{by.get('exact', 0)} / {by.get('fuzzy', 0)} / {by.get('none', 0)}"
        ),
    }
    partes: list[str] = []
    if inventados:
        partes.append(
            f"{len(inventados)} de {fid.n} trechos não existem no texto enviado — tratados como "
            f"inventados: {', '.join(inventados)}."
        )
    if sem_ficha:
        partes.append(
            f"{len(sem_ficha)} extração(ões) sem ficha válida, sem trecho a verificar: "
            f"{', '.join(sem_ficha)}."
        )
    conclusao = " ".join(partes) or (
        f"Todos os {fid.n} trechos foram localizados no texto enviado ao modelo."
    )
    return AuditBlock("a) Fidelidade", numeros, conclusao)


def stability_block(summary: AuditSummary) -> AuditBlock:
    """4.4b — duas execuções idênticas."""
    st = summary.stability
    if st is None:
        raise ValueError("Relatório exige a verificação de estabilidade (4.4b).")
    d = st.diff
    numeros: dict[str, Any] = {
        "pares comparados": d.n_pairs,
        "fichas idênticas": f"{d.n_identical}/{d.n_pairs} ({pct(d.rate_identical)})",
        "campos que mais mudam": _top_fields(d.field_change_rate),
    }
    if d.n_unpaired:
        # Sem ficha válida num dos lados: fora do denominador, mas nunca escondido.
        numeros["sem par comparável"] = d.n_unpaired
    worst = st.most_unstable_fields()
    if worst and worst[0][1] > 0:
        conclusao = (
            f"Sem mudar nada, {d.n_pairs - d.n_identical} de {d.n_pairs} fichas mudaram; "
            f"o campo mais instável é '{worst[0][0]}' ({pct(worst[0][1])} dos artigos)."
        )
    else:
        conclusao = f"As {d.n_pairs} fichas foram idênticas nas duas execuções."
    return AuditBlock("b) Estabilidade", numeros, conclusao)


def input_block(summary: AuditSummary) -> AuditBlock:
    """4.4c — duas estratégias de seleção."""
    ie = summary.input_effect
    if ie is None:
        raise ValueError("Relatório exige a verificação de efeito da entrada (4.4c).")
    d = ie.diff
    verdict = ie.recommend()
    numeros = {
        "estratégias": f"{ie.strategy_a} × {ie.strategy_b}",
        "fichas idênticas": f"{d.n_identical}/{d.n_pairs}"
        + (f" (+{d.n_unpaired} sem par)" if d.n_unpaired else ""),
        "divergências substantivas": len(ie.disagreements()),
        "fidelidade": (
            f"{ie.strategy_a} {pct(ie.fidelity_a.rate_found)} · "
            f"{ie.strategy_b} {pct(ie.fidelity_b.rate_found)}"
        ),
        "campos que mais divergem": _top_fields(d.field_change_rate),
    }
    conclusao = f"Defendemos {verdict.strategy}: {verdict.explanation}"
    return AuditBlock("c) Efeito da entrada", numeros, conclusao)


def temperature_block(summary: AuditSummary) -> AuditBlock:
    """4.4d — outra temperatura num subconjunto."""
    tr = summary.temperature
    if tr is None:
        raise ValueError("Relatório exige a verificação de temperatura (4.4d).")
    t0, ta = temp(tr.temperature_t0), temp(tr.temperature_alt)
    numeros = {
        "subconjunto": f"{len(tr.arquivos)} artigos",
        "JSON válido de primeira": (
            f"t={t0}: {pct(tr.rate_json_valid_first_try_t0)} · "
            f"t={ta}: {pct(tr.rate_json_valid_first_try_alt)}"
        ),
        "fidelidade": f"t={t0}: {pct(tr.rate_fidelity_t0)} · t={ta}: {pct(tr.rate_fidelity_alt)}",
        "fichas idênticas": f"{tr.diff.n_identical}/{tr.diff.n_pairs}",
    }
    if tr.escolha_t0_se_sustenta:
        conclusao = f"A escolha de t={t0} se sustenta: t={ta} não melhora validade nem fidelidade."
    else:
        conclusao = (
            f"A escolha de t={t0} NÃO se sustenta pelo critério declarado: t={ta} teve validade "
            "ou fidelidade maior neste subconjunto — reportado como está."
        )
    return AuditBlock("d) Efeito da temperatura", numeros, conclusao)


def _num(value: Any) -> float | None:
    """Célula numérica de um ``DataFrame`` como ``float`` (``None`` se ausente)."""
    return None if value is None or pd.isna(value) else float(value)


def _int(value: float | None) -> int | str:
    return "—" if value is None else int(value)


def prompt_table(comparison: PromptComparison) -> pd.DataFrame:
    """Tabela compacta da comparação de prompts (taxas em %, rótulos em português)."""
    frame = comparison.to_frame()
    a, b = comparison.variants
    rows: list[dict[str, Any]] = []
    for key, label in _PROMPT_ROWS.items():
        if key not in frame.index:
            continue
        va, vb = _num(frame.loc[key, a]), _num(frame.loc[key, b])
        if key.startswith("rate_"):
            rows.append({"métrica": label, a: pct(va), b: pct(vb)})
        else:
            rows.append({"métrica": label, a: _int(va), b: _int(vb)})
    return pd.DataFrame(rows, columns=["métrica", a, b])


def prompt_conclusion(comparison: PromptComparison) -> str:
    """Vencedor pela regra declarada + onde as fichas discordam."""
    verdict = comparison.decide()
    dis = comparison.disagreements()
    campos = sorted({d.field for d in dis})
    onde = f" Discordam em {len(dis)} campo(s): {', '.join(campos)}." if dis else ""
    return f"{verdict.explanation}{onde} Versão final: {verdict.winner}."


def cost_section(
    comparison: CostComparison, per_run: Sequence[CostReport] | None = None
) -> CostSection:
    """Tabela de custo (por execução + total real + ingênuo) com a premissa visível."""
    rows: list[dict[str, Any]] = []
    for rep in per_run or []:
        rows.append(_cost_row(rep.label, rep))
    rows.append(_cost_row("TOTAL real (todas as execuções)", comparison.real))
    rows.append(_cost_row(comparison.naive.label, comparison.naive))
    return CostSection(
        tabela=rows,
        premissa=comparison.real.premise.describe(),
        razao_ingenuo_real=comparison.ratio,
        comentario=(
            f"Tokens de entrada: a alternativa ingênua envia {comparison.token_ratio:.1f}× mais. "
            "Repetições de estabilidade, a outra estratégia, a outra temperatura e a outra "
            "versão do prompt estão incluídas no total real."
        ),
    )


def _cost_row(label: str, rep: CostReport) -> dict[str, Any]:
    return {
        "cenário": label,
        "chamadas": rep.n_calls,
        "tokens entrada": rep.input_tokens,
        "tokens saída": rep.output_tokens,
        "US$": f"{rep.usd_total:.4f}".replace(".", ","),
    }


def nao_defensaveis_table(summary: AuditSummary) -> pd.DataFrame:
    """``arquivo, motivos`` das fichas com confiança BAIXA."""
    return pd.DataFrame(
        [
            {"arquivo": f.arquivo, "motivos": "; ".join(f.motivos)}
            for f in summary.nao_defensaveis()
        ],
        columns=["arquivo", "motivos"],
    )


def build_report_content(
    *,
    integrantes: Sequence[str],
    modelo: str,
    summary: AuditSummary,
    cost: CostComparison,
    prompt_comparison: PromptComparison | None = None,
    cost_per_run: Sequence[CostReport] | None = None,
    strategy_table: pd.DataFrame | None = None,
    narrative: Narrative | None = None,
    figuras: Sequence[Path] = (),
) -> ReportContent:
    """Monta o conteúdo completo do relatório a partir dos resultados medidos.

    Args:
        integrantes: nomes completos (primeira página).
        modelo: modelo declarado (ex.: ``"Qwen/Qwen2.5-3B-Instruct (float16, T4)"``).
        summary: resumo da auditoria com as quatro verificações preenchidas.
        cost: comparação real × ingênuo (4.5).
        prompt_comparison: comparação das duas versões do prompt (4.2); padrão:
            ``summary.prompt_comparison``.
        cost_per_run: custo por execução (opcional, detalha a tabela).
        strategy_table: resumo das estratégias de seleção (opcional).
        narrative: textos de justificativa; padrão: resumo dos ADRs.
        figuras: PNGs opcionais (entram só se couberem nas 3 páginas).

    """
    nar = narrative or Narrative()
    prompt_comparison = prompt_comparison or summary.prompt_comparison
    if prompt_comparison is None:
        raise ValueError("Relatório exige a comparação das duas versões do prompt (4.2).")
    over = nar.conclusoes
    a, b = prompt_comparison.variants
    blocks = AuditResults(
        fidelidade=fidelity_block(summary),
        estabilidade=stability_block(summary),
        entrada=input_block(summary),
        temperatura=temperature_block(summary),
    )
    for key in ("fidelidade", "estabilidade", "entrada", "temperatura"):
        if key in over:
            getattr(blocks, key).conclusao = over[key]
    dist = summary.confidence_distribution
    regra = (
        f"{summary.rule.describe()}\nDistribuição resultante: alta {dist['alta']}, "
        f"média {dist['media']}, baixa {dist['baixa']}."
    )
    custo = cost_section(cost, cost_per_run)
    if "custo" in over:
        custo.comentario = over["custo"]
    return ReportContent(
        titulo=nar.titulo,
        integrantes=list(integrantes),
        estrategia_escolhida=nar.estrategia_escolhida,
        estrategia_justificativa=nar.estrategia_justificativa,
        estrategia_tabela=strategy_table,
        prompt_v1=PromptVersion(a, features_label(a), "Versão de referência."),
        prompt_v2=PromptVersion(b, features_label(b), "Difere da referência em uma técnica."),
        prompt_diff=describe_diff(a, b),
        prompt_comparacao=prompt_table(prompt_comparison),
        prompt_conclusao=over.get("prompt", prompt_conclusion(prompt_comparison)),
        modelo=modelo,
        modelo_justificativa=nar.modelo_justificativa,
        temperatura_justificativa=nar.temperatura_justificativa,
        resultados_auditoria=blocks,
        regra_confianca=regra,
        fichas_nao_defensaveis=nao_defensaveis_table(summary),
        custo=custo,
        o_que_faria_diferente=list(nar.o_que_faria_diferente),
        declaracao_uso_ia=nar.declaracao_uso_ia,
        figuras=list(figuras),
    )
