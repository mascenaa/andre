"""Resumo da auditoria: junta as quatro verificações, a comparação de prompts e a confiança.

É o objeto que o notebook exibe e o relatório consome: :meth:`AuditSummary.to_dict` (números
de manchete) e :meth:`AuditSummary.tables` (DataFrames prontos). :func:`fichas_nao_defensaveis`
responde "quais fichas você não defenderia e por quê".
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ficha.audit.confidence import ConfidenceRule, FichaAuditada, build_final_fichas
from ficha.audit.fidelity import FidelitySummary, fidelity_summary
from ficha.audit.input_effect import InputEffectReport, input_effect_report
from ficha.audit.leakage import LeakageSummary, leakage_summary
from ficha.audit.prompt_compare import PromptComparison, compare_prompt_variants
from ficha.audit.stability import StabilityReport, stability_report
from ficha.audit.temperature import TemperatureReport, temperature_report
from ficha.schema import Confianca
from ficha.types import ExtractionRecord


def fichas_nao_defensaveis(fichas: Sequence[FichaAuditada]) -> list[FichaAuditada]:
    """Fichas com confiança BAIXA (cada uma com seus motivos), na ordem recebida."""
    return [f for f in fichas if f.confianca == Confianca.BAIXA]


def confidence_distribution(fichas: Sequence[FichaAuditada]) -> dict[str, int]:
    """``{"alta": n, "media": n, "baixa": n}`` (sempre com as três chaves)."""
    return {c.value: sum(f.confianca == c for f in fichas) for c in Confianca}


def confidence_by_rule(
    primary: Sequence[ExtractionRecord],
    stability_rep: Sequence[ExtractionRecord] | None,
    alt_input: Sequence[ExtractionRecord] | None,
    rules: Mapping[str, ConfidenceRule],
) -> pd.DataFrame:
    """Confiança por artigo sob várias regras, lado a lado (ex.: ``{"v1": RULE_V1, "v2": ...}``).

    Colunas: ``arquivo``, e para cada regra ``confianca_<nome>`` e ``motivos_<nome>``.
    A distribuição de cada regra sai de ``df[f"confianca_{nome}"].value_counts()`` ou de
    :func:`confidence_distribution` aplicada às fichas.
    """
    df = pd.DataFrame({"arquivo": [r.arquivo for r in primary]})
    for name, rule in rules.items():
        fichas = build_final_fichas(primary, stability_rep, alt_input, rule)
        df[f"confianca_{name}"] = [f.confianca.value for f in fichas]
        df[f"motivos_{name}"] = ["; ".join(f.motivos) for f in fichas]
    return df


@dataclass(frozen=True, slots=True)
class AuditSummary:
    """Tudo o que a Seção 4.4 (e a medição da 4.2) produziu, num só lugar.

    Verificações opcionais (``None``) simplesmente não geram tabela.
    """

    fidelity: FidelitySummary
    fichas: list[FichaAuditada]
    rule: ConfidenceRule = field(default_factory=ConfidenceRule)
    stability: StabilityReport | None = None
    input_effect: InputEffectReport | None = None
    temperature: TemperatureReport | None = None
    prompt_comparison: PromptComparison | None = None
    leakage: LeakageSummary | None = None
    """4.4e: vazamento de exemplos few-shot na execução primária."""

    @property
    def confidence_distribution(self) -> dict[str, int]:
        return confidence_distribution(self.fichas)

    def nao_defensaveis(self) -> list[FichaAuditada]:
        return fichas_nao_defensaveis(self.fichas)

    def to_dict(self) -> dict[str, Any]:
        """Números de manchete (serializáveis em JSON)."""
        return {
            "regra_confianca": self.rule.describe(),
            "regra_versao": self.rule.version,
            "regra_input_effect_mode": self.rule.input_effect_mode,
            "fidelidade": self.fidelity.to_dict(),
            "estabilidade": self.stability.to_dict() if self.stability else None,
            "efeito_entrada": self.input_effect.to_dict() if self.input_effect else None,
            "temperatura": self.temperature.to_dict() if self.temperature else None,
            "comparacao_prompts": (
                self.prompt_comparison.to_dict() if self.prompt_comparison else None
            ),
            "vazamento_exemplos": self.leakage.to_dict() if self.leakage else None,
            "distribuicao_confianca": self.confidence_distribution,
            "nao_defensaveis": [
                {"arquivo": f.arquivo, "motivos": list(f.motivos)} for f in self.nao_defensaveis()
            ],
        }

    def tables(self) -> dict[str, pd.DataFrame]:
        """DataFrames nomeados; chaves presentes quando a verificação existe.

        - ``fidelidade``: uma linha por ficha (``arquivo, run_id, found, score, method,
          page_claimed, page_found, page_ok, threshold``).
        - ``estabilidade_campos`` / ``entrada_campos`` / ``temperatura_campos``:
          ``field, change_rate, mean_similarity``.
        - ``entrada_divergencias``: ``arquivo, field, a, b, equal, similarity`` (sim < 0.8).
        - ``entrada_estrategias``, ``temperatura``, ``prompts``: métricas lado a lado.
        - ``vazamento``: uma linha por campo copiado de exemplo few-shot (``arquivo, run_id,
          field, example_index, example_name, similarity, method, value, example_value``).
        - ``confianca``: ``confianca, n``.
        - ``fichas``: a tabela final com colunas ``audit_*``.
        - ``nao_defensaveis``: ``arquivo, confianca, motivos``.
        """
        out: dict[str, pd.DataFrame] = {"fidelidade": self.fidelity.to_frame()}
        if self.stability:
            out["estabilidade_campos"] = self.stability.field_frame()
        if self.input_effect:
            out["entrada_campos"] = self.input_effect.diff.field_frame()
            out["entrada_estrategias"] = self.input_effect.to_frame()
            div = self.input_effect.disagreements()
            out["entrada_divergencias"] = pd.DataFrame(
                [d.to_dict() for d in div],
                columns=["arquivo", "field", "a", "b", "equal", "similarity"],
            )
        if self.temperature:
            out["temperatura"] = self.temperature.to_frame()
            out["temperatura_campos"] = self.temperature.diff.field_frame()
        if self.prompt_comparison:
            out["prompts"] = self.prompt_comparison.to_frame()
            out["prompts_campos"] = self.prompt_comparison.diff.field_frame()
        if self.leakage:
            out["vazamento"] = self.leakage.to_frame()
        dist = self.confidence_distribution
        out["confianca"] = pd.DataFrame(
            {"confianca": list(dist), "n": list(dist.values())}, columns=["confianca", "n"]
        )
        out["fichas"] = pd.DataFrame([f.to_row() for f in self.fichas])
        out["nao_defensaveis"] = pd.DataFrame(
            [
                {
                    "arquivo": f.arquivo,
                    "confianca": f.confianca.value,
                    "motivos": "; ".join(f.motivos),
                }
                for f in self.nao_defensaveis()
            ],
            columns=["arquivo", "confianca", "motivos"],
        )
        return out


def build_audit_summary(
    primary: Sequence[ExtractionRecord],
    *,
    stability_rep: Sequence[ExtractionRecord] | None = None,
    alt_input: Sequence[ExtractionRecord] | None = None,
    t_alt: Sequence[ExtractionRecord] | None = None,
    prompt_runs: tuple[Sequence[ExtractionRecord], Sequence[ExtractionRecord]] | None = None,
    rule: ConfidenceRule | None = None,
    limitacao_gabarito: Mapping[str, bool] | None = None,
) -> AuditSummary:
    """Roda todas as verificações disponíveis e monta o :class:`AuditSummary`.

    - ``primary``: execução escolhida (1 registro por artigo) — base da tabela final.
    - ``stability_rep``: repetição da mesma configuração (4.4b).
    - ``alt_input``: mesma extração com a outra estratégia de entrada (4.4c).
    - ``t_alt``: subconjunto com a temperatura alternativa (4.4d), comparado com ``primary``.
    - ``prompt_runs``: ``(variante A, variante B)`` da medição obrigatória da 4.2.

    O vazamento de exemplos few-shot (4.4e) é sempre calculado sobre ``primary``.
    """
    rule = rule or ConfidenceRule()
    th = rule.fidelity_threshold
    return AuditSummary(
        fidelity=fidelity_summary(primary, th),
        fichas=build_final_fichas(primary, stability_rep, alt_input, rule),
        rule=rule,
        leakage=leakage_summary(primary, threshold=rule.leakage_threshold),
        stability=stability_report(primary, stability_rep) if stability_rep is not None else None,
        input_effect=(
            input_effect_report(
                primary, alt_input, threshold=th, limitacao_gabarito=limitacao_gabarito
            )
            if alt_input is not None
            else None
        ),
        temperature=temperature_report(primary, t_alt, threshold=th) if t_alt is not None else None,
        prompt_comparison=(
            compare_prompt_variants(
                prompt_runs[0], prompt_runs[1], threshold=th, limitacao_gabarito=limitacao_gabarito
            )
            if prompt_runs is not None
            else None
        ),
    )
