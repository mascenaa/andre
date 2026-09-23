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
from ficha.audit.confidence import CALIBRACAO_ENTRADA
from ficha.audit.normalize import normalize_for_match
from ficha.cost import CostComparison, CostReport
from ficha.prompts import FEW_SHOT_EXAMPLES, VARIANTS, describe_diff
from ficha.report.content import (
    AuditBlock,
    AuditResults,
    CostSection,
    PromptVersion,
    ReportContent,
)
from ficha.types import Document

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
            "Anotar à mão um gabarito de limitacao em 5 artigos, para separar abstenção "
            "correta de omissão (hoje indistinguíveis).",
            'Instruir explicitamente "não copie os exemplos" no prompt e medir se o '
            "vazamento do few-shot para a evidência desaparece.",
            "Comparar o modelo local com um modelo por API nas mesmas 19 fichas.",
        ]
    )
    cobertura_limitacao: str = ""
    """Evidência medida sobre a seleção (ex.: quantas frases de limitação do texto completo
    chegam ao contexto de cada estratégia). Vai para a seção 1 do relatório."""
    nota_limitacao: str = ""
    """Frase sobre ``limitacao`` null (gabarito, heurística), anexada à conclusão de 4.4c."""
    destaque: str | None = None
    """Sobrescreve o parágrafo de destaque gerado por :func:`invention_highlight`."""
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


def short_name(arquivo: str) -> str:
    """Rótulo curto de um artigo: ``08_Leka_2019_Comparison...pdf`` → ``08_Leka_2019``."""
    stem = arquivo.removesuffix(".pdf")
    parts = stem.split("_")
    return "_".join(parts[:3]) if len(parts) > 3 else stem


def few_shot_leak(trecho: str) -> str | None:
    """Nome do exemplo few-shot de onde ``trecho`` foi copiado, ou ``None``.

    Um trecho que existe no exemplo do prompt, mas não no artigo, é a invenção mais
    convincente possível: frase real, bem formada, com cara de artigo científico.
    """
    needle = normalize_for_match(trecho)
    if len(needle) < 20:
        return None
    for ex in FEW_SHOT_EXAMPLES:
        if needle in normalize_for_match(ex.context_text) or needle in normalize_for_match(
            ex.output.evidencia.trecho
        ):
            return ex.name
    return None


def _trechos(summary: AuditSummary) -> dict[str, str]:
    return {f.arquivo: f.ficha.evidencia.trecho for f in summary.fichas if not f.failed}


def fidelity_block(summary: AuditSummary) -> AuditBlock:
    """4.4a — quantos trechos existem no texto enviado, e com a página certa.

    ``failures()`` mistura dois casos que o relatório separa: trecho devolvido mas ausente do
    texto enviado (invenção) e extração sem ficha (``method == "none"``, nada a verificar).
    Trechos encontrados em página diferente da declarada são listados à parte.
    """
    fid = summary.fidelity
    trechos = _trechos(summary)
    inventados = [r for r in fid.failures() if r.method != "none"]
    sem_ficha = [r.arquivo for r in fid.failures() if r.method == "none"]
    pagina_errada = [r for r in fid.results if r.found and not r.page_ok]
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
        itens = []
        for r in inventados:
            leak = few_shot_leak(trechos.get(r.arquivo, ""))
            origem = " (cópia do exemplo few-shot do prompt)" if leak else ""
            itens.append(f"{short_name(r.arquivo)}{origem}")
        partes.append(
            f"{len(inventados)}/{fid.n} trecho(s) inventado(s), ausente(s) do texto enviado: "
            f"{', '.join(itens)}."
        )
    if pagina_errada:
        itens = [
            f"{short_name(r.arquivo)}: declarada {r.page_claimed}, encontrada {r.page_found}"
            for r in pagina_errada
        ]
        partes.append(
            f"Página errada em {len(pagina_errada)}/{fid.n} ({'; '.join(itens)}): trecho real, "
            "mas a ficha não é rastreável como declarada — BAIXA."
        )
    if sem_ficha:
        partes.append(
            f"{len(sem_ficha)} extração(ões) sem ficha válida, sem trecho a verificar: "
            f"{', '.join(short_name(a) for a in sem_ficha)}."
        )
    conclusao = " ".join(partes) or (
        f"Todos os {fid.n} trechos foram localizados no texto enviado, na página declarada."
    )
    return AuditBlock("a) Fidelidade", numeros, conclusao)


def invention_highlight(summary: AuditSummary) -> str:
    """Parágrafo de destaque sobre a invenção mais convincente (o que a rubrica pede).

    Prioriza um trecho copiado do exemplo few-shot; senão, o primeiro trecho não encontrado.
    Vazio se não houve trecho inventado.
    """
    trechos = _trechos(summary)
    by_file = {f.arquivo: f for f in summary.fichas}
    inventados = [r for r in summary.fidelity.failures() if r.method != "none"]
    if not inventados:
        return ""
    leaks = [(r, few_shot_leak(trechos.get(r.arquivo, ""))) for r in inventados]
    r, leak = next(((r, lk) for r, lk in leaks if lk), leaks[0])
    trecho = trechos.get(r.arquivo, "")
    formato = (
        "O JSON era válido de primeira e a página declarada, plausível: nenhuma checagem de "
        "formato o pegaria. "
        if by_file.get(r.arquivo) is not None and by_file[r.arquivo].parse_status.value == "ok"
        else ""
    )
    if leak:
        origem = (
            "a frase do EXEMPLO few-shot do prompt, ausente do artigo. Detectado porque a "
            "fidelidade compara o trecho com o texto efetivamente enviado ao modelo, e o exemplo "
            "não faz parte dele"
        )
    else:
        origem = (
            "uma frase que não existe no texto enviado. Detectado porque a fidelidade compara o "
            "trecho com o texto efetivamente enviado ao modelo, não com o que parece plausível"
        )
    return (
        f"Invenção convincente — {short_name(r.arquivo)}: o modelo devolveu como evidência "
        f"«{trecho}», {origem} (similaridade {format_sim(r.score)} < limiar "
        f"{format_sim(r.threshold)}). "
        f"{formato}A ficha foi rebaixada a BAIXA."
    )


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
        conclusao = (
            f"{d.n_identical}/{d.n_pairs} fichas idênticas nas duas execuções — esperado com "
            "decodificação gulosa e seed fixa. Por isso a variabilidade real do pipeline aparece "
            "no efeito da entrada (c) e da temperatura (d), não aqui."
        )
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
        "limitacao null": (
            f"{ie.strategy_a} {_count(ie.stats_a.rate_limitacao_null, ie.stats_a.n)} · "
            f"{ie.strategy_b} {_count(ie.stats_b.rate_limitacao_null, ie.stats_b.n)}"
        ),
    }
    conclusao = f"Defendemos {verdict.strategy}: {verdict.explanation}"
    return AuditBlock("c) Efeito da entrada", numeros, conclusao)


def _count(rate: float, n: int) -> str:
    """``0.947, 19`` → ``"18/19"``."""
    return f"{round(rate * n)}/{n}"


TEXT_FIELDS: tuple[str, ...] = ("problema", "dados", "metodo", "metrica", "evidencia.trecho")
"""Campos de texto livre usados na calibração da regra de confiança."""


def mean_text_similarity(similarities: Mapping[str, float]) -> float | None:
    """Média da similaridade dos campos de texto livre (``TEXT_FIELDS``) de um ``DiffReport``."""
    vals = [similarities[f] for f in TEXT_FIELDS if f in similarities]
    vals = [v for v in vals if not pd.isna(v)]
    return sum(vals) / len(vals) if vals else None


def calibration_sentence(
    summary: AuditSummary, distributions: Mapping[str, Mapping[str, int]] | None = None
) -> str:
    """Por que a regra final compara estratégias só por discordância categórica.

    Números: similaridade média dos campos de texto livre entre estratégias × entre repetições,
    e (se dadas) as distribuições de confiança de cada versão da regra.
    """
    partes: list[str] = []
    if distributions:
        dist_txt = "; ".join(
            f"{nome}: alta {d.get('alta', 0)}/média {d.get('media', 0)}/baixa {d.get('baixa', 0)}"
            for nome, d in distributions.items()
        )
        partes.append(f"Calibração da regra — {dist_txt}.")
    entre_estrategias = (
        mean_text_similarity(summary.input_effect.diff.field_mean_similarity)
        if summary.input_effect
        else None
    )
    entre_repeticoes = (
        mean_text_similarity(summary.stability.diff.field_mean_similarity)
        if summary.stability
        else None
    )
    if entre_estrategias is not None and entre_repeticoes is not None:
        partes.append(
            "A similaridade média dos campos de texto livre é "
            f"{format_sim(entre_estrategias)} entre estratégias (paráfrase de entradas "
            f"diferentes) e {format_sim(entre_repeticoes)} entre repetições: comparar o texto "
            "exato entre estratégias mede redação, não confiabilidade. Por isso a regra final só "
            "conta discordância categórica entre estratégias (limitação declarada × null)."
        )
    return " ".join(partes)


def format_sim(value: float) -> str:
    """Similaridade 0–1 com duas casas e vírgula decimal."""
    return f"{value:.2f}".replace(".", ",")


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
            f"Pelo critério declarado, t={t0} não se sustenta neste subconjunto: t={ta} teve "
            f"fidelidade {pct(tr.rate_fidelity_alt)} vs {pct(tr.rate_fidelity_t0)} em t={t0}."
        )
        erros_t0 = _t0_errors(summary, tr.arquivos)
        if erros_t0:
            conclusao += f" Erro(s) em t={t0}: {erros_t0}."
    n = len(tr.arquivos)
    if n < SMALL_SUBSET:
        estab = summary.stability.diff if summary.stability else None
        manter = (
            f" Mantemos t={t0} pela estabilidade ({estab.n_identical}/{estab.n_pairs} fichas "
            "idênticas entre repetições)"
            if estab is not None
            else f" Mantemos t={t0}"
        )
        conclusao += (
            f" Com n={n} artigos, a diferença não sustenta conclusão.{manter} e reportamos o "
            "resultado como está."
        )
    return AuditBlock("d) Efeito da temperatura", numeros, conclusao)


SMALL_SUBSET = 10
"""Abaixo deste número de artigos, a comparação de temperatura é reportada como indicativa."""


def _t0_errors(summary: AuditSummary, arquivos: Sequence[str]) -> str:
    """Falhas de fidelidade da execução principal dentro do subconjunto, com a causa."""
    trechos = _trechos(summary)
    subset = set(arquivos)
    out = []
    for r in summary.fidelity.failures():
        if r.arquivo not in subset:
            continue
        leak = few_shot_leak(trechos.get(r.arquivo, ""))
        out.append(short_name(r.arquivo) + (" (vazamento do exemplo few-shot)" if leak else ""))
    return ", ".join(out)


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


def final_pages_share(docs: Sequence[Document], last_fraction: float = 1 / 3) -> float | None:
    """Fração das frases de limitação declarada que ficam no último terço de cada artigo."""
    from ficha.select import limitation_sentences

    total = final = 0
    for d in docs:
        cut = d.n_pages * (1 - last_fraction)
        for page, _ in limitation_sentences(d):
            total += 1
            final += page > cut
    return final / total if total else None


def coverage_sentence(recall: pd.DataFrame, n_docs: int, final_share: float | None = None) -> str:
    """Frase da seção 1 a partir da tabela ``ficha.select.recall_table`` (sem números à mão)."""
    if recall.empty:
        return ""
    first = recall.to_dict("records")[0]
    rows = recall.to_dict("records")
    por_estrategia = ", ".join(
        f"{r['estrategia']} {int(r['frases_no_contexto'])} ({pct(_num(r['recall_frases']))}; "
        f"{int(r['artigos_cobertos'])}/{int(r['artigos_com_frases'])} artigos)"
        for r in rows
    )
    onde = (
        f" Só {pct(final_share)} delas estão no último terço do artigo: estão espalhadas, e "
        "nenhuma estratégia de orçamento fixo cobre a maioria."
        if final_share is not None
        else ""
    )
    return (
        f"Das {int(first['frases_total'])} frases em que os autores declaram limitação "
        f"({int(first['artigos_com_frases'])} dos {n_docs} artigos têm ao menos uma), chegaram ao "
        f"contexto enviado: {por_estrategia}.{onde} O null em limitacao é sobretudo efeito da "
        "seleção (o modelo não declara o que não recebe), não da abstenção do modelo."
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
    rule_distributions: Mapping[str, Mapping[str, int]] | None = None,
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
        rule_distributions: distribuições de confiança por versão da regra (ex.:
            ``{"v1 estrita": {...}, "v2 categórica (final)": {...}}``) para a calibração.

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
    if nar.nota_limitacao:
        blocks.entrada.conclusao += f" {nar.nota_limitacao}"
    dist = summary.confidence_distribution
    # O texto fixo de calibração da regra é um instantâneo; no relatório vale o calculado
    # a partir destes dados (``calibration_sentence``), para não haver dois números.
    regra_txt = summary.rule.describe().replace(CALIBRACAO_ENTRADA, "").rstrip()
    regra = (
        f"{regra_txt}\nDistribuição resultante: alta {dist['alta']}, "
        f"média {dist['media']}, baixa {dist['baixa']}."
    )
    calibracao = calibration_sentence(summary, rule_distributions)
    if calibracao:
        regra += f"\n{calibracao}"
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
        estrategia_evidencia=nar.cobertura_limitacao,
        destaque_auditoria=(
            nar.destaque if nar.destaque is not None else invention_highlight(summary)
        ),
    )
