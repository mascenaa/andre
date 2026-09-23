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

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ficha.audit import AuditSummary, PromptComparison
from ficha.audit.confidence import CALIBRACAO_ENTRADA
from ficha.audit.fidelity import FidelitySummary, fidelity_summary
from ficha.audit.leakage import LeakageSummary, leakage_summary
from ficha.cost import CostComparison, CostReport
from ficha.prompts import VARIANTS, describe_diff
from ficha.report.content import (
    AuditBlock,
    AuditResults,
    CostSection,
    PromptVersion,
    ReportContent,
)
from ficha.types import Document, ExtractionRecord

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
    "rate_fewshot_leak": "vazamento de exemplos",
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
            "Uma segunda passagem dedicada só a limitacao, sobre as janelas de pista de "
            "limitação (a entrada já as traz; o modelo de 3B não as usa na ficha completa).",
            "Modelo maior — Qwen2.5-7B em 4 bits ou um modelo por API — nas mesmas 19 fichas.",
            'O prompt já diz "não copie o conteúdo dos exemplos" e isso não impediu o vazamento; '
            "testar exemplos-esqueleto (campos com marcadores em vez de valores) e medir se o "
            "vazamento some sem perder a fidelidade que o few-shot trouxe.",
            "Gabarito manual de limitacao em 5 artigos, para separar abstenção de omissão.",
        ]
    )
    cobertura_limitacao: str = ""
    """Evidência medida sobre a seleção (ex.: quantas frases de limitação do texto completo
    chegam ao contexto de cada estratégia). Vai para a seção 1 do relatório."""
    antes_depois: str = ""
    """Frase "antes × depois" da seleção (ver :func:`before_after_sentence`), anexada a 4.4c."""
    nota_limitacao: str = ""
    """Frase sobre ``limitacao`` null (gabarito, heurística), anexada à conclusão de 4.4c."""
    nota_temperatura: str = ""
    """Frase anexada à conclusão de 4.4d (ex.: como o subconjunto foi escolhido)."""
    nota_custo: str = ""
    """Frase anexada ao comentário do custo (ex.: por que uma execução tem menos chamadas)."""
    nota_nao_defensaveis: str = ""
    """Parágrafo após a tabela de fichas não defensáveis (o que a regra não vê e a leitura viu)."""
    destaques_preferidos: list[str] = field(default_factory=list)
    """Prefixos de arquivo a destacar primeiro (ex.: ``["08_Leka", "06_Barnes"]``)."""
    destaque: str | None = None
    """Sobrescreve o parágrafo de destaque gerado por :func:`invention_highlight`."""
    declaracao_uso_ia: str = (
        "Usamos assistentes de IA para escrever partes do código e revisar textos. As decisões "
        "de projeto, os números e as conclusões foram verificados pelo grupo."
    )
    titulo: str = "Atividade de Construção I — Ficha comparativa de 19 artigos"
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


def _leak_map(leakage: LeakageSummary | None) -> dict[str, list[str]]:
    """``{arquivo: campos copiados dos exemplos few-shot}`` (fonte: ``ficha.audit.leakage``)."""
    if leakage is None:
        return {}
    return {r.arquivo: r.leaked_fields for r in leakage.leaks()}


def fidelity_block(summary: AuditSummary) -> AuditBlock:
    """4.4a — quantos trechos existem no texto enviado, e com a página certa.

    ``failures()`` mistura dois casos que o relatório separa: trecho devolvido mas ausente do
    texto enviado (invenção) e extração sem ficha (``method == "none"``, nada a verificar).
    Trechos encontrados em página diferente da declarada são listados à parte. Quando o trecho
    inventado foi copiado do exemplo few-shot (``summary.leakage``), isso é dito.
    """
    fid = summary.fidelity
    leaks = _leak_map(summary.leakage)
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
        itens = [
            short_name(r.arquivo)
            + (
                " (cópia do exemplo few-shot do prompt)"
                if "evidencia.trecho" in leaks.get(r.arquivo, [])
                else ""
            )
            for r in inventados
        ]
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


@dataclass(frozen=True, slots=True)
class InventionCase:
    """Uma invenção convincente: campos copiados do exemplo e/ou trecho fora do texto enviado."""

    execucao: str
    arquivo: str
    leaked_fields: list[str]
    example_name: str | None
    values: dict[str, str]
    """Valor devolvido pelo modelo em cada campo copiado."""
    copied_fields: list[str]
    """Campos copiados por inteiro (similaridade com o campo do exemplo acima do limiar)."""
    terms: dict[str, str]
    """Contaminação parcial: ``{campo: termo do exemplo ausente do texto enviado}``."""
    trecho: str
    trecho_found: bool
    page_ok: bool
    json_ok: bool


def invention_cases(
    runs: Mapping[str, Sequence[ExtractionRecord]],
) -> list[InventionCase]:
    """Casos de invenção em cada execução (vazamento few-shot ou trecho não encontrado).

    Um artigo aparece uma vez só: na primeira execução (na ordem de ``runs``) em que o caso ocorre.
    """
    out: list[InventionCase] = []
    vistos: set[str] = set()
    for label, recs in runs.items():
        leak: LeakageSummary = leakage_summary(recs)
        fid: FidelitySummary = fidelity_summary(recs)
        by_leak = {r.arquivo: r for r in leak.leaks()}
        by_fid = {r.arquivo: r for r in fid.results}
        for rec in recs:
            if rec.ficha is None or rec.arquivo in vistos:
                continue
            lr = by_leak.get(rec.arquivo)
            fr = by_fid[rec.arquivo]
            hits = [*lr.leaked, *lr.leaked_terms] if lr else []
            if lr is None and fr.found:
                continue
            vistos.add(rec.arquivo)
            out.append(
                InventionCase(
                    execucao=label,
                    arquivo=rec.arquivo,
                    leaked_fields=lr.leaked_fields if lr else [],
                    example_name=hits[0].example_name if hits else None,
                    values={h.field: h.value for h in hits},
                    copied_fields=[h.field for h in lr.leaked] if lr else [],
                    terms={h.field: h.example_value for h in lr.leaked_terms} if lr else {},
                    trecho=rec.ficha.evidencia.trecho,
                    trecho_found=fr.found,
                    page_ok=fr.page_ok,
                    json_ok=rec.json_valid_first_try,
                )
            )
    return out


def _clip(text: str, n: int = 110) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _kind(c: InventionCase) -> int:
    """Tipo de detecção de um caso (menor = mais sutil).

    0 = campo copiado com trecho correto (só o vazamento pega); 1 = cópia com trecho fora do
    texto enviado; 2 = trecho inventado sem cópia do exemplo.
    """
    if c.leaked_fields and c.trecho_found:
        return 0
    return 1 if c.leaked_fields else 2


def pick_highlights(
    cases: Sequence[InventionCase], k: int = 2, prefer: Sequence[str] = ()
) -> list[InventionCase]:
    """Até ``k`` casos em destaque.

    Primeiro os preferidos (prefixo do arquivo, na ordem dada), depois um de cada tipo de
    detecção, depois o resto.
    """
    chosen: list[InventionCase] = []
    for prefixo in prefer:
        achado = next((c for c in cases if c.arquivo.startswith(prefixo)), None)
        if achado is not None and achado not in chosen and len(chosen) < k:
            chosen.append(achado)
    for kind in (1, 0, 2):
        found = next((c for c in cases if _kind(c) == kind), None)
        if found is not None and found not in chosen and len(chosen) < k:
            chosen.append(found)
    for c in cases:
        if len(chosen) >= k:
            break
        if c not in chosen:
            chosen.append(c)
    return chosen


def invention_highlight(
    cases: Sequence[InventionCase], k: int = 2, prefer: Sequence[str] = ()
) -> str:
    """Parágrafo de destaque (o que a rubrica pede): as invenções e por que foram detectadas.

    Mostra até ``k`` casos (um por tipo de detecção); os demais são contados, não descritos.
    """
    if not cases:
        return ""
    shown = pick_highlights(cases, k, prefer)
    partes: list[str] = []
    for i, c in enumerate(shown, start=1):
        nome = f"({i}) {short_name(c.arquivo)}, execução {c.execucao}"
        if c.leaked_fields and not c.trecho_found:
            campos = ", ".join(f for f in c.leaked_fields if f != "evidencia.trecho")
            outros = f"os campos {campos} e " if campos else ""
            partes.append(
                f"{nome}: o modelo copiou do exemplo few-shot do prompt {outros}o trecho "
                f"«{_clip(c.trecho)}», ausente do artigo. Detectado pela fidelidade, que compara o "
                "trecho com o texto enviado — do qual o exemplo não faz parte."
            )
        elif c.leaked_fields:
            campo = (c.copied_fields or c.leaked_fields)[0]
            if campo in c.copied_fields:
                como = "foi copiado do exemplo few-shot"
            else:
                como = f"contém «{c.terms.get(campo, '')}», termo do exemplo ausente do artigo"
            partes.append(
                f"{nome}: o campo {campo} «{_clip(c.values.get(campo, ''))}» {como}, com trecho de "
                "evidência correto — fidelidade e página passam. Só a comparação dos campos com "
                "os exemplos do prompt detecta."
            )
        else:
            partes.append(
                f"{nome}: trecho «{_clip(c.trecho)}» ausente do texto enviado. Detectado pela "
                "fidelidade, que compara com o texto efetivamente enviado ao modelo."
            )
    formato = (
        " JSON válido de primeira: nenhuma checagem de formato os pegaria."
        if all(c.json_ok for c in shown)
        else ""
    )
    resto = len(cases) - len(shown)
    outros = f" Outros {resto} caso(s) nas tabelas das verificações a) e e)." if resto else ""
    return "Invenções convincentes detectadas — " + " ".join(partes) + formato + outros


def leakage_block(runs: Mapping[str, Sequence[ExtractionRecord]]) -> AuditBlock:
    """4.4e — campos copiados dos exemplos few-shot, por execução."""
    numeros: dict[str, Any] = {}
    total = 0
    for label, recs in runs.items():
        ls = leakage_summary(recs)
        total += ls.n_with_leak
        campos = [f for f, n in ls.by_field.items() if n]
        numeros[label] = f"{ls.n_with_leak}/{ls.n}" + (f" ({', '.join(campos)})" if campos else "")
    conclusao = (
        "Cópias do exemplo aparecem mesmo com exemplos em domínio propositalmente distante; "
        "a verificação compara cada campo com o campo homólogo dos exemplos (limiar 0,80)."
        if total
        else "Nenhum campo reproduz os exemplos do prompt."
    )
    return AuditBlock("e) Vazamento de exemplos few-shot", numeros, conclusao)


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
    n = len(tr.arquivos)
    numeros = {
        "subconjunto": f"{n} artigos",
        "JSON válido de primeira": (
            f"t={t0}: {pct(tr.rate_json_valid_first_try_t0)} · "
            f"t={ta}: {pct(tr.rate_json_valid_first_try_alt)}"
        ),
        "fidelidade": f"t={t0}: {pct(tr.rate_fidelity_t0)} · t={ta}: {pct(tr.rate_fidelity_alt)}",
        "página correta": (
            f"t={t0}: {_count(tr.stats_t0.rate_page_ok, n)} · "
            f"t={ta}: {_count(tr.stats_alt.rate_page_ok, n)}"
        ),
        "fichas idênticas": f"{tr.diff.n_identical}/{tr.diff.n_pairs}",
    }
    if tr.escolha_t0_se_sustenta:
        conclusao = f"A escolha de t={t0} se sustenta: t={ta} não melhora validade nem fidelidade."
    else:
        conclusao = (
            f"Pelo critério declarado (JSON válido, depois fidelidade), t={t0} perde neste "
            f"subconjunto: fidelidade {pct(tr.rate_fidelity_t0)} vs {pct(tr.rate_fidelity_alt)} "
            f"em t={ta}."
        )
        erros_t0 = _t0_errors(summary, tr.arquivos)
        if erros_t0:
            conclusao += f" A diferença vem de: {erros_t0}."
        if tr.stats_t0.rate_page_ok > tr.stats_alt.rate_page_ok:
            conclusao += (
                " Já a página correta, o critério que decidiu a 4.4c, favorece "
                f"t={t0}: {_count(tr.stats_t0.rate_page_ok, n)} vs "
                f"{_count(tr.stats_alt.rate_page_ok, n)}; pela regra de confiança, t={ta} "
                "geraria mais fichas BAIXA."
            )
    if n < SMALL_SUBSET:
        conclusao += (
            f" Com n={n} artigos nada disso é conclusivo; mantemos t={t0} pela página correta e "
            "porque a saída determinística deixa a estabilidade (b) sem ruído de amostragem, e "
            "reportamos o resultado como está."
        )
    return AuditBlock("d) Efeito da temperatura", numeros, conclusao)


SMALL_SUBSET = 10
"""Abaixo deste número de artigos, a comparação de temperatura é reportada como indicativa."""


def _t0_errors(summary: AuditSummary, arquivos: Sequence[str]) -> str:
    """Falhas de fidelidade da execução principal dentro do subconjunto, com a causa."""
    leaks = _leak_map(summary.leakage)
    subset = set(arquivos)
    out = []
    for r in summary.fidelity.failures():
        if r.arquivo not in subset:
            continue
        causa = " (vazamento do exemplo few-shot)" if leaks.get(r.arquivo) else ""
        out.append(short_name(r.arquivo) + causa)
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
    """Vencedor pela regra declarada + onde as fichas discordam + o trade-off do few-shot."""
    verdict = comparison.decide()
    dis = comparison.disagreements()
    campos = sorted({d.field for d in dis})
    onde = f" Discordam em {len(dis)} campo(s): {', '.join(campos)}." if dis else ""
    texto = f"{verdict.explanation}{onde} Versão final: {verdict.winner}."
    frame = comparison.to_frame()
    a, b = comparison.variants
    if "rate_fewshot_leak" in frame.index:
        la, lb = _num(frame.loc["rate_fewshot_leak", a]), _num(frame.loc["rate_fewshot_leak", b])
        fa, fb = _num(frame.loc["rate_fidelity", a]), _num(frame.loc["rate_fidelity", b])
        if la is not None and lb is not None and la != lb:
            com, sem = (a, b) if la > lb else (b, a)
            lc, ls_ = max(la, lb), min(la, lb)
            fc, fs = (fa, fb) if com == a else (fb, fa)
            texto += (
                f" Trade-off: {com} tem fidelidade {pct(fc)} × {pct(fs)}, mas {pct(lc)} das "
                f"fichas copiam campos do exemplo (× {pct(ls_)} em {sem}); o critério de vitória "
                "declarado não muda."
            )
    return texto


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
            "O total real inclui tudo o que foi enviado: repetição de estabilidade, outra "
            "estratégia, outra temperatura, outra versão do prompt e os experimentos de seleção "
            "(execuções hybrid)."
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


_LEAK_COPY = re.compile(r"^campo (\S+) copiado do exemplo few-shot \(similaridade ([\d.]+)\)$")
_LEAK_TERMS = re.compile(
    r"^campo (\S+) contém (.+?), termos? do exemplo few-shot ausentes? do texto enviado$"
)


def compact_motivos(motivos: Sequence[str]) -> list[str]:
    """Junta os motivos de vazamento (um por campo) em dois, para a tabela do PDF caber.

    A lista completa continua na coluna ``motivos_confianca`` do CSV/XLSX.
    """
    copies: list[tuple[str, float]] = []
    terms: list[tuple[str, str]] = []
    rest: list[str] = []
    for m in motivos:
        if mc := _LEAK_COPY.match(m):
            copies.append((mc.group(1), float(mc.group(2))))
        elif mt := _LEAK_TERMS.match(m):
            terms.append((mt.group(1), mt.group(2)))
        else:
            rest.append(m)
    out: list[str] = []
    if copies:
        sims = [s for _, s in copies]
        faixa = f"{min(sims):.2f}" if min(sims) == max(sims) else f"{min(sims):.2f}–{max(sims):.2f}"
        campos = ", ".join(c for c, _ in copies)
        plural = "s" if len(copies) > 1 else ""
        out.append(
            f"campo{plural} {campos} copiado{plural} do exemplo few-shot (similaridade {faixa})"
        )
    if terms:
        vistos: list[str] = []
        for _, t in terms:
            vistos += [x.strip() for x in t.split(",") if x.strip() not in vistos]
        out.append(
            f"termos do exemplo few-shot ausentes do texto enviado em "
            f"{', '.join(c for c, _ in terms)}: {', '.join(dict.fromkeys(vistos))}"
        )
    return out + rest


def nao_defensaveis_table(summary: AuditSummary) -> pd.DataFrame:
    """``arquivo, motivos`` das fichas com confiança BAIXA (motivos de vazamento compactados)."""
    return pd.DataFrame(
        [
            {"arquivo": f.arquivo, "motivos": "; ".join(compact_motivos(f.motivos))}
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


def _milhar(value: int) -> str:
    """Inteiro com ponto de milhar (``2861`` → ``"2.861"``)."""
    return f"{value:,}".replace(",", ".")


def budget_sentence(
    curve: pd.DataFrame,
    recall: pd.DataFrame,
    docs: Sequence[Document],
    chosen: str = "hybrid 12k",
    alternative: str = "hybrid 20k",
    baseline: str = "semantic",
) -> str:
    """Curva de orçamento do ``hybrid`` e a decisão de custo, a partir das tabelas medidas."""
    rows = {str(r["estrategia"]): r for r in curve.to_dict("records")}
    base_rows = {str(r["estrategia"]): r for r in recall.to_dict("records")}
    pontos = " · ".join(
        f"{nome.removeprefix('hybrid ')}: {pct(_num(r['recall_frases']))}, "
        f"{int(r['artigos_cobertos'])}/{int(r['artigos_com_frases'])} artigos, "
        f"~{_milhar(int(r['tokens_por_artigo']))} tok"
        for nome, r in rows.items()
    )
    texto = f"Curva de orçamento do hybrid (recall das frases; artigos; tokens/artigo) — {pontos}."
    if chosen in rows and docs:
        escolhido = rows[chosen]
        artigo_medio = sum(d.n_chars for d in docs) / len(docs)
        mais_barato = artigo_medio / float(escolhido["chars_medios"])
        extra = ""
        if baseline in base_rows:
            base_chars = float(base_rows[baseline]["chars_medios"])
            extra = (
                f" e +{pct(float(escolhido['chars_medios']) / base_chars - 1)} de caracteres "
                f"sobre o {baseline}"
            )
        vezes = f"{mais_barato:.1f}".replace(".", ",")
        texto += (
            f" Escolhemos {chosen.removeprefix('hybrid ')} por custo: {vezes}× mais barato que "
            f"o artigo inteiro{extra}."
        )
        if alternative in rows:
            alt = rows[alternative]
            mais_tokens = float(alt["tokens_por_artigo"]) / float(escolhido["tokens_por_artigo"])
            texto += (
                f" {alternative.removeprefix('hybrid ')} fica documentado como alternativa "
                f"({int(alt['artigos_cobertos'])}/{int(alt['artigos_com_frases'])} artigos, "
                f"+{pct(mais_tokens - 1)} de tokens)."
            )
    return texto


def limitation_outcome(
    runs: Mapping[str, Sequence[ExtractionRecord]], threshold: float | None = None
) -> pd.DataFrame:
    """Por execução: quantas fichas têm ``limitacao`` preenchida, e quantas destas com trecho fiel.

    É a validação a jusante da seleção (ADR 0002): mais frases de limitação no contexto só
    importam se viram ``limitacao`` preenchida **com evidência verificável** (4.4a).
    """
    from ficha.audit import check_fidelity
    from ficha.audit.fidelity import DEFAULT_FIDELITY_THRESHOLD

    th = DEFAULT_FIDELITY_THRESHOLD if threshold is None else threshold
    rows = []
    for nome, recs in runs.items():
        preenchidas = [r for r in recs if r.ficha is not None and r.ficha.limitacao is not None]
        fieis = [r for r in preenchidas if check_fidelity(r, th).found]
        rows.append(
            {
                "execucao": nome,
                "n": len(recs),
                "limitacao_preenchida": len(preenchidas),
                "preenchida_com_trecho_fiel": len(fieis),
                "limitacao_null": len(recs) - len(preenchidas),
            }
        )
    return pd.DataFrame(rows)


def before_after_sentence(outcome: pd.DataFrame, before: str, after: str) -> str:
    """Frase "antes × depois" da seleção para ``limitacao`` (diz se os nulls não caíram)."""
    rows = {str(r["execucao"]): r for r in outcome.to_dict("records")}
    if before not in rows or after not in rows:
        return ""
    a, b = rows[before], rows[after]
    texto = (
        f"Antes × depois da seleção: limitacao null {int(a['limitacao_null'])}/{int(a['n'])} com "
        f"{before} → {int(b['limitacao_null'])}/{int(b['n'])} com {after}; preenchidas com trecho "
        f"fiel: {int(a['preenchida_com_trecho_fiel'])} → {int(b['preenchida_com_trecho_fiel'])}."
    )
    if int(b["limitacao_null"]) >= int(a["limitacao_null"]):
        texto += f" O {after} não reduziu os nulls."
    return texto


SENTINELAS_LIMITACAO = frozenset(
    {
        "não informado",
        "nao informado",
        "não informada",
        "nao informada",
        "não declarado",
        "nao declarado",
        "não declarada",
        "nao declarada",
        "n/a",
        "null",
        "none",
        "",
    }
)
"""Textos que significam "ausente" escritos no lugar do ``null`` (não são limitação)."""


def _is_sentinel(value: str) -> bool:
    return value.strip().strip(".").lower() in SENTINELAS_LIMITACAO


def strategy_comparison(
    runs: Mapping[str, Sequence[ExtractionRecord]],
    recall: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """As estratégias lado a lado, a partir das execuções gravadas (uma por estratégia).

    ``limitacao_legitima`` = preenchida, que não é sentinela ("Não informado") nem cópia do
    exemplo few-shot. É o que a troca de estratégia queria aumentar.
    """
    from ficha.audit import check_fewshot_leakage

    recall_by = (
        {str(r["estrategia"]): _num(r["recall_frases"]) for r in recall.to_dict("records")}
        if recall is not None
        else {}
    )
    rows = []
    for nome, recs in runs.items():
        fid = fidelity_summary(recs)
        leaks = {r.arquivo: check_fewshot_leakage(r) for r in recs}
        preenchidas = [r for r in recs if r.ficha is not None and r.ficha.limitacao is not None]
        sentinela = [r for r in preenchidas if _is_sentinel(r.ficha.limitacao or "")]  # type: ignore[union-attr]
        vazada = [r for r in preenchidas if "limitacao" in leaks[r.arquivo].leaked_fields]
        legitima = [r for r in preenchidas if r not in sentinela and r not in vazada]
        estrategia = recs[0].strategy if recs else nome
        rows.append(
            {
                "estrategia": nome,
                "fichas_ok": sum(r.ficha is not None for r in recs),
                "n": len(recs),
                "fidelidade": fid.n_found,
                "pagina_correta": fid.n_page_ok,
                "vazamento": sum(lr.any_leak for lr in leaks.values()),
                "limitacao_preenchida": len(preenchidas),
                "limitacao_legitima": len(legitima),
                "limitacao_vazada": len(vazada),
                "limitacao_sentinela": len(sentinela),
                "tokens_entrada": sum(r.usage.input_tokens for r in recs),
                "recall_limitacoes_no_contexto": recall_by.get(estrategia),
                "legitimas": ", ".join(short_name(r.arquivo) for r in legitima),
            }
        )
    return pd.DataFrame(rows)


def strategy_verdict(
    comparison: pd.DataFrame,
    chosen: str,
    experiment: str,
    recall: pd.DataFrame | None = None,
    criterion: str = "pagina_correta",
) -> str:
    """Conclusão da 4.1/4.4c: o experimento de entrada mudou a saída? (a partir das tabelas)."""
    rows = {str(r["estrategia"]): r for r in comparison.to_dict("records")}
    if chosen not in rows or experiment not in rows:
        return ""
    a, b = rows[chosen], rows[experiment]
    n = int(a["n"])
    entrada = ""
    if recall is not None:
        rr = {str(r["estrategia"]): r for r in recall.to_dict("records")}
        ka, kb = chosen.split(" ")[0], experiment.split(" ")[0]
        if ka in rr and kb in rr:
            entrada = (
                f"Corrigimos a entrada — artigos com limitação declarada no contexto: "
                f"{int(rr[ka]['artigos_cobertos'])} → {int(rr[kb]['artigos_cobertos'])} de "
                f"{int(rr[ka]['artigos_com_frases'])}; recall das frases "
                f"{pct(_num(rr[ka]['recall_frases']))} → {pct(_num(rr[kb]['recall_frases']))}. "
            )
    tokens = float(b["tokens_entrada"]) / float(a["tokens_entrada"]) - 1
    sinal = "+" if tokens >= 0 else ""
    saida = (
        f"limitacao legítima {int(a['limitacao_legitima'])}/{n} ({a['legitimas'] or '—'}) × "
        f"{int(b['limitacao_legitima'])}/{n} ({b['legitimas'] or '—'}); no {experiment}, "
        f"{int(b['limitacao_vazada'])} preenchida(s) copiada(s) do exemplo e "
        f"{int(b['limitacao_sentinela'])} sentinela"
    )
    if int(b["limitacao_legitima"]) <= int(a["limitacao_legitima"]):
        efeito = (
            f"e a saída não mudou: {saida}. O gargalo de limitacao é o modelo, não a seleção; o "
            "próximo passo é um modelo maior (ou API) ou uma segunda passagem dedicada só a "
            "limitação."
        )
    else:
        efeito = f"e a saída melhorou: {saida}."
    return (
        f"{entrada}{efeito[0].upper()}{efeito[1:]} Pelo critério "
        f"declarado da 4.4c ({criterion.replace('_', ' ')}: {int(a[criterion])}/{n} × "
        f"{int(b[criterion])}/{n}; tokens de entrada {sinal}{pct(tokens)} no {experiment}), a "
        f"estratégia final é {chosen.split(' ')[0]}."
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
    leak_runs: Mapping[str, Sequence[ExtractionRecord]] | None = None,
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
        leak_runs: execuções rotuladas (a principal primeiro) para o bloco e) de vazamento e o
            destaque das invenções; sem elas, nenhum dos dois entra no relatório.

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
        vazamento=leakage_block(leak_runs) if leak_runs else None,
    )
    for key in ("fidelidade", "estabilidade", "entrada", "temperatura", "vazamento"):
        if key in over and getattr(blocks, key) is not None:
            getattr(blocks, key).conclusao = over[key]
    for extra in (nar.antes_depois, nar.nota_limitacao):
        if extra:
            blocks.entrada.conclusao += f" {extra}"
    if nar.nota_temperatura:
        blocks.temperatura.conclusao += f" {nar.nota_temperatura}"
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
    if nar.nota_custo:
        custo.comentario += f" {nar.nota_custo}"
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
        nota_nao_defensaveis=nar.nota_nao_defensaveis,
        custo=custo,
        o_que_faria_diferente=list(nar.o_que_faria_diferente),
        declaracao_uso_ia=nar.declaracao_uso_ia,
        figuras=list(figuras),
        estrategia_evidencia=nar.cobertura_limitacao,
        destaque_auditoria=(
            nar.destaque
            if nar.destaque is not None
            else invention_highlight(
                invention_cases(leak_runs or {}), prefer=nar.destaques_preferidos
            )
        ),
    )
