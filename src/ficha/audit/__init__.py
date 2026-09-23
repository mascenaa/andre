"""Auditoria (Seção 4.4) e medição de prompts (Seção 4.2).

- :mod:`~ficha.audit.normalize`      — normalização tipográfica (premissa das comparações).
- :mod:`~ficha.audit.fidelity`       — 4.4a: o trecho existe no texto enviado? Página certa?
- :mod:`~ficha.audit.diff`           — motor de comparação entre duas execuções.
- :mod:`~ficha.audit.stability`      — 4.4b: repetição sem mudar nada.
- :mod:`~ficha.audit.input_effect`   — 4.4c: duas estratégias de entrada.
- :mod:`~ficha.audit.temperature`    — 4.4d: outra temperatura num subconjunto.
- :mod:`~ficha.audit.prompt_compare` — 4.2: duas versões do prompt e o vencedor por regra.
- :mod:`~ficha.audit.confidence`     — regra declarada de ``confianca`` e fichas finais.
- :mod:`~ficha.audit.summary`        — tudo junto, em tabelas.
"""

from ficha.audit.confidence import ConfidenceRule, FichaAuditada, build_final_fichas
from ficha.audit.diff import FIELDS, DiffReport, FieldDiff, diff_runs
from ficha.audit.fidelity import (
    FidelityResult,
    FidelitySummary,
    LimitacaoCheck,
    check_fidelity,
    check_limitacao_support,
    fidelity_summary,
)
from ficha.audit.input_effect import InputEffectReport, input_effect_report
from ficha.audit.normalize import normalize_for_match, split_context_pages
from ficha.audit.prompt_compare import PromptComparison, PromptVerdict, compare_prompt_variants
from ficha.audit.runstats import RunStats, run_stats
from ficha.audit.stability import StabilityReport, stability_report
from ficha.audit.summary import (
    AuditSummary,
    build_audit_summary,
    confidence_distribution,
    fichas_nao_defensaveis,
)
from ficha.audit.temperature import TemperatureReport, temperature_report

__all__ = [
    "FIELDS",
    "AuditSummary",
    "ConfidenceRule",
    "DiffReport",
    "FichaAuditada",
    "FidelityResult",
    "FidelitySummary",
    "FieldDiff",
    "InputEffectReport",
    "LimitacaoCheck",
    "PromptComparison",
    "PromptVerdict",
    "RunStats",
    "StabilityReport",
    "TemperatureReport",
    "build_audit_summary",
    "build_final_fichas",
    "check_fidelity",
    "check_limitacao_support",
    "compare_prompt_variants",
    "confidence_distribution",
    "diff_runs",
    "fichas_nao_defensaveis",
    "fidelity_summary",
    "input_effect_report",
    "normalize_for_match",
    "run_stats",
    "split_context_pages",
    "stability_report",
    "temperature_report",
]
