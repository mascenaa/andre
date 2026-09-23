"""Regra declarada de atribuição de ``confianca`` (Seção 4.4, parágrafo final).

``confianca`` nunca vem do modelo nem "do olho": é função determinística do que as
verificações mediram. A regra completa está em :meth:`ConfidenceRule.describe`, cujo texto vai
literalmente para o relatório. Cada ficha recebe também a lista de **motivos** que a
impediram de ter nível mais alto — é o que sustenta "quais fichas você não defenderia".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from ficha.audit.diff import DiffReport, diff_runs, index_by_arquivo
from ficha.audit.fidelity import (
    DEFAULT_FIDELITY_THRESHOLD,
    FidelityResult,
    LimitacaoCheck,
    check_fidelity,
    check_limitacao_support,
)
from ficha.schema import Confianca, Evidencia, Ficha
from ficha.types import ExtractionRecord, ParseStatus

InputEffectMode = Literal["strict", "categorical"]

CALIBRACAO_ENTRADA = (
    "Calibração após a execução real (Qwen2.5-3B, 19 artigos, 2026-09-23): entre as "
    "estratégias semantic e first_pages a similaridade média dos campos de texto livre foi "
    "≈0.49 (problema 0.49, dados 0.46, metodo 0.49, metrica 0.49, evidencia.trecho 0.48; "
    "evidencia.pagina 0.16), enquanto entre repetições com a mesma entrada foi 1.00 (19/19 "
    "fichas idênticas). Entradas diferentes produzem paráfrases e evidências de páginas "
    "diferentes por construção: isso mede redação, não confiabilidade."
)
"""Justificativa numérica do modo ``categorical`` (citada em :meth:`ConfidenceRule.describe`)."""

_ORDER = {Confianca.BAIXA: 0, Confianca.MEDIA: 1, Confianca.ALTA: 2}

FALHOU = "EXTRAÇÃO FALHOU"
"""Texto dos campos de uma ficha cuja extração falhou (a tabela mantém as 19 linhas)."""


def _min(a: Confianca, b: Confianca) -> Confianca:
    return a if _ORDER[a] <= _ORDER[b] else b


@dataclass(frozen=True, slots=True)
class ConfidenceRule:
    """Parâmetros da regra de confiança. Os padrões são a regra usada no relatório."""

    fidelity_threshold: float = DEFAULT_FIDELITY_THRESHOLD
    """Score mínimo (``partial_ratio``/100) para o trecho contar como encontrado."""
    max_changed_fields_for_alta: int = 0
    """Máximo de campos que podem mudar (em cada comparação) para ainda ser ALTA."""
    max_changed_fields_for_media: int = 2
    """Máximo de campos que podem mudar para ainda ser MEDIA; acima disso, BAIXA."""
    require_page_ok: bool = True
    """Se ``True``, trecho encontrado em página diferente da declarada → BAIXA."""
    check_limitacao: bool = True
    """Se ``True``, limitação preenchida sem vocabulário de limitação no contexto → máx. MEDIA.

    Só o veredito ``preenchida_sem_suporte`` rebaixa. ``null_suspeito`` (null, mas o contexto
    contém a palavra "limitation") **nunca** rebaixa: na execução real deu falsos positivos
    confirmados por leitura (um título "Prospects, Scope, and Limitations"; um parágrafo sobre
    "limitations of generative AI in academic writing") — a palavra aparece sem ser limitação
    declarada do próprio trabalho.
    """
    input_effect_mode: InputEffectMode = "categorical"
    """Como contar "campos que mudaram" entre ESTRATÉGIAS de entrada (4.4c).

    - ``"strict"`` (regra v1): igualdade exata após normalização em todos os campos.
    - ``"categorical"`` (regra v2, padrão): só conta a discordância categórica em
      ``limitacao`` (null de um lado, texto do outro). Ver :data:`CALIBRACAO_ENTRADA`.

    Entre REPETIÇÕES (mesma entrada) a igualdade é sempre exata, nos dois modos.
    """

    @property
    def version(self) -> str:
        """``"v1"`` (strict) ou ``"v2"`` (categorical), para rotular tabelas lado a lado."""
        return "v1" if self.input_effect_mode == "strict" else "v2"

    def describe(self) -> str:
        """Texto da regra, em português, para o relatório (depende do modo ativo)."""
        a, m = self.max_changed_fields_for_alta, self.max_changed_fields_for_media
        pag = (
            ", ou o trecho foi encontrado numa página diferente da declarada em evidencia.pagina"
            if self.require_page_ok
            else ""
        )
        strict = self.input_effect_mode == "strict"
        if strict:
            baixa_cmp = f"mais de {m} campos mudaram entre as duas repetições ou entre as duas "
            baixa_cmp += "estratégias de entrada"
            media_cmp = (
                f"entre {a + 1} e {m} campos mudaram em alguma das comparações (repetição ou "
                "estratégia)"
            )
            alta_cmp = (
                f"no máximo {a} campo(s) mudou(aram) tanto entre repetições quanto entre "
                "estratégias"
            )
            modo = (
                "Modo de comparação entre estratégias: strict (v1) — todo campo diferente após "
                "normalização conta como mudança, também entre estratégias de entrada."
            )
        else:
            baixa_cmp = f"mais de {m} campos mudaram entre as duas repetições (mesma entrada)"
            media_cmp = (
                f"entre {a + 1} e {m} campos mudaram entre as repetições, ou limitacao é null "
                "numa estratégia de entrada e preenchida na outra (discordância categórica)"
            )
            alta_cmp = (
                f"no máximo {a} campo(s) mudou(aram) entre repetições e não há discordância "
                "categórica entre estratégias"
            )
            modo = (
                "Modo de comparação entre estratégias: categorical (v2) — entre estratégias só "
                "conta como mudança a discordância categórica em limitacao (null de um lado, "
                "texto do outro); paráfrase de texto livre e evidência de outra página não "
                "contam. Entre repetições (mesma entrada) a comparação continua exata. "
                f"{CALIBRACAO_ENTRADA}"
            )
        lim = (
            "\n- No máximo MEDIA se o campo limitacao veio preenchido mas o texto enviado não "
            "contém nenhum termo de limitação (limitation, shortcoming, drawback, caveat, future "
            "work, threats to validity...), sinal de limitação possivelmente inventada. "
            "O caso inverso (limitacao null com a palavra 'limitation' no contexto) NÃO "
            "rebaixa: a heurística teve falsos positivos (títulos de seção e uso genérico da "
            "palavra)."
            if self.check_limitacao
            else ""
        )
        return (
            f"Regra de confiança {self.version} (aplicada automaticamente; o nível final é o "
            "MENOR entre os limites abaixo):\n"
            "- BAIXA se a extração falhou (nenhum JSON válido segundo o esquema), "
            "ou o trecho de evidência não foi encontrado no texto enviado ao modelo "
            f"(similaridade partial_ratio normalizada < {self.fidelity_threshold:.2f})"
            f"{pag}, ou {baixa_cmp}.\n"
            f"- MEDIA se a evidência passou, mas {media_cmp}, ou se alguma comparação não pôde "
            "ser feita (ausente ou sem ficha válida do outro lado): o que não foi verificado "
            f"não recebe ALTA.{lim}\n"
            f"- ALTA somente se a evidência foi encontrada na página declarada e {alta_cmp}.\n"
            "Campos comparados: problema, dados, metodo, metrica, limitacao, evidencia.trecho, "
            "evidencia.pagina; textos iguais após normalização tipográfica (NFKC, minúsculas, "
            f"espaços, aspas e hífens).\n{modo}"
        )

    def assign(
        self,
        fidelity: FidelityResult,
        n_changed_fields_stability: int | None,
        n_changed_fields_input: int | None,
        parse_status: ParseStatus,
        limitacao_supported: bool | None = None,
    ) -> tuple[Confianca, list[str]]:
        """Nível de confiança e os motivos que o limitaram.

        ``None`` numa comparação significa "não verificado" e limita a MEDIA.
        ``n_changed_fields_input`` deve vir contado conforme ``input_effect_mode`` (é o que
        :func:`build_final_fichas` faz). ``limitacao_supported=False`` (veredito
        ``preenchida_sem_suporte``) limita a MEDIA quando ``check_limitacao`` está ligado;
        ``True``/``None`` não têm efeito — ``null_suspeito`` conta como ``True``.
        Para ALTA, a lista traz uma única frase com o que foi confirmado.
        """
        level = Confianca.ALTA
        reasons: list[str] = []

        def cap(to: Confianca, reason: str) -> None:
            nonlocal level
            level = _min(level, to)
            reasons.append(reason)

        if parse_status == ParseStatus.FAILED:
            cap(
                Confianca.BAIXA,
                "extração falhou: o modelo não devolveu ficha válida (parse FAILED)",
            )
            return level, reasons

        found = fidelity.method != "none" and fidelity.score >= self.fidelity_threshold
        if not found:
            cap(
                Confianca.BAIXA,
                f"trecho não encontrado no texto enviado (score {fidelity.score:.2f} < "
                f"{self.fidelity_threshold:.2f}): possível trecho inventado",
            )
        elif self.require_page_ok and not fidelity.page_ok:
            onde = f"p. {fidelity.page_found}" if fidelity.page_found else "página indeterminada"
            cap(
                Confianca.BAIXA,
                f"página errada: declarada p. {fidelity.page_claimed}, trecho localizado em {onde}",
            )

        for nome, n in (
            ("repetições", n_changed_fields_stability),
            ("estratégias de entrada", n_changed_fields_input),
        ):
            categorical = nome != "repetições" and self.input_effect_mode == "categorical"
            if categorical and n is not None and n > self.max_changed_fields_for_alta:
                lvl = Confianca.BAIXA if n > self.max_changed_fields_for_media else Confianca.MEDIA
                cap(
                    lvl,
                    "discordância categórica entre estratégias de entrada: limitacao é null "
                    "numa e preenchida na outra",
                )
                continue
            if n is None:
                cap(Confianca.MEDIA, f"comparação entre {nome} não verificada")
            elif n > self.max_changed_fields_for_media:
                cap(Confianca.BAIXA, f"{n} campos mudaram entre {nome} (instável)")
            elif n > self.max_changed_fields_for_alta:
                cap(Confianca.MEDIA, f"{n} campo(s) mudou(aram) entre {nome}")

        if self.check_limitacao and limitacao_supported is False:
            cap(
                Confianca.MEDIA,
                "limitação preenchida, mas o texto enviado não tem vocabulário de limitação: "
                "possível limitação inventada",
            )

        if level == Confianca.ALTA:
            reasons = [
                f"trecho encontrado ({fidelity.method}, score {fidelity.score:.2f}) na página "
                f"declarada p. {fidelity.page_claimed}; estável entre repetições e estratégias"
            ]
        return level, reasons


@dataclass(frozen=True, slots=True)
class FichaAuditada:
    """Ficha final (com ``confianca``) e a trilha de auditoria que a justifica."""

    ficha: Ficha
    motivos: list[str]
    fidelity: FidelityResult
    n_changed_stability: int | None
    n_changed_input: int | None
    parse_status: ParseStatus = ParseStatus.OK
    run_id: str = ""
    limitacao: LimitacaoCheck | None = None
    diffs_stability: list[str] = field(default_factory=list)
    """Campos que mudaram entre repetições."""
    diffs_input: list[str] = field(default_factory=list)
    """Campos que mudaram entre estratégias (no modo ``categorical``, só os categóricos)."""

    @property
    def arquivo(self) -> str:
        return self.ficha.arquivo

    @property
    def confianca(self) -> Confianca:
        assert self.ficha.confianca is not None
        return self.ficha.confianca

    @property
    def failed(self) -> bool:
        return self.parse_status == ParseStatus.FAILED

    def to_row(self) -> dict[str, Any]:
        """Linha da tabela final: campos da ficha + colunas de auditoria (prefixo ``audit_``)."""
        row = self.ficha.to_row()
        if self.failed:
            row["evidencia_pagina"] = None
        row.update(
            {
                "audit_motivos": "; ".join(self.motivos),
                "audit_fidelity_score": self.fidelity.score,
                "audit_fidelity_method": self.fidelity.method,
                "audit_page_found": self.fidelity.page_found,
                "audit_page_ok": self.fidelity.page_ok,
                "audit_n_changed_stability": self.n_changed_stability,
                "audit_n_changed_input": self.n_changed_input,
                "audit_campos_instaveis": ", ".join(self.diffs_stability),
                "audit_campos_divergentes_entrada": ", ".join(self.diffs_input),
                "audit_limitacao": self.limitacao.verdict if self.limitacao else None,
                "audit_parse_status": self.parse_status.value,
                "audit_run_id": self.run_id,
            }
        )
        return row


def failed_ficha(arquivo: str) -> Ficha:
    """Ficha placeholder para extração que falhou (mantém a linha na tabela, com BAIXA)."""
    return Ficha(
        arquivo=arquivo,
        problema=FALHOU,
        dados=FALHOU,
        metodo=FALHOU,
        metrica=FALHOU,
        limitacao=None,
        evidencia=Evidencia(trecho=f"{FALHOU}: sem evidência devolvida pelo modelo", pagina=1),
        confianca=Confianca.BAIXA,
    )


def _changes(
    report: DiffReport | None, arquivo: str, categorical: bool = False
) -> tuple[int | None, list[str]]:
    if report is None:
        return None, []
    if categorical:
        return report.n_categorical_changes(arquivo), report.categorical_changed_fields(arquivo)
    return report.n_changed(arquivo), report.changed_fields(arquivo)


def build_final_fichas(
    primary: Sequence[ExtractionRecord],
    stability_rep: Sequence[ExtractionRecord] | None,
    alt_input: Sequence[ExtractionRecord] | None,
    rule: ConfidenceRule | None = None,
) -> list[FichaAuditada]:
    """Monta as fichas finais com ``confianca`` segundo ``rule``.

    - ``primary``: a execução escolhida (um registro por artigo; ordem preservada).
    - ``stability_rep``: a repetição da MESMA configuração (4.4b), ou ``None``.
    - ``alt_input``: a mesma extração com a outra estratégia de entrada (4.4c), ou ``None``.

    ``n_changed_input`` é contado conforme ``rule.input_effect_mode`` (exato em ``strict``,
    só discordâncias categóricas de ``limitacao`` em ``categorical``).

    Registros com extração falha viram fichas ``EXTRAÇÃO FALHOU`` com ``BAIXA``, para que a
    tabela tenha sempre uma linha por artigo.
    """
    rule = rule or ConfidenceRule()
    index_by_arquivo(primary, "primária")  # falha alto se houver duplicata
    stab = diff_runs(primary, stability_rep) if stability_rep is not None else None
    inp = diff_runs(primary, alt_input) if alt_input is not None else None

    out: list[FichaAuditada] = []
    for rec in primary:
        fid = check_fidelity(rec, rule.fidelity_threshold)
        n_stab, f_stab = _changes(stab, rec.arquivo)
        n_inp, f_inp = _changes(inp, rec.arquivo, rule.input_effect_mode == "categorical")
        lim = check_limitacao_support(rec)
        # Sem ficha é falha, qualquer que seja o status gravado.
        status = ParseStatus.FAILED if rec.ficha is None else rec.parse_status
        level, motivos = rule.assign(fid, n_stab, n_inp, status, lim.supported)
        if rec.ficha is None:
            ficha = failed_ficha(rec.arquivo)
        else:
            ficha = Ficha.from_extraida(rec.arquivo, rec.ficha, confianca=level)
        out.append(
            FichaAuditada(
                ficha=ficha,
                motivos=motivos,
                fidelity=fid,
                n_changed_stability=n_stab,
                n_changed_input=n_inp,
                parse_status=status,
                run_id=rec.run_id,
                limitacao=lim,
                diffs_stability=f_stab,
                diffs_input=f_inp,
            )
        )
    return out


RULE_V1 = ConfidenceRule(input_effect_mode="strict")
"""Regra v1 (antes da calibração): comparação exata também entre estratégias."""
RULE_V2 = ConfidenceRule(input_effect_mode="categorical")
"""Regra v2 (padrão): entre estratégias só conta a discordância categórica em limitacao."""
