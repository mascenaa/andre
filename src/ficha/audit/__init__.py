"""Auditoria (Seção 4.4) e medição de prompts (Seção 4.2).

- :mod:`~ficha.audit.normalize`      — normalização tipográfica (premissa das comparações).
- :mod:`~ficha.audit.fidelity`       — 4.4a: o trecho existe no texto enviado? Página certa?
- :mod:`~ficha.audit.diff`           — motor de comparação entre duas execuções.
- :mod:`~ficha.audit.stability`      — 4.4b: repetição sem mudar nada.
- :mod:`~ficha.audit.input_effect`   — 4.4c: duas estratégias de entrada.
- :mod:`~ficha.audit.temperature`    — 4.4d: outra temperatura num subconjunto.
- :mod:`~ficha.audit.leakage`        — 4.4e: campos copiados dos exemplos few-shot.
- :mod:`~ficha.audit.prompt_compare` — 4.2: duas versões do prompt e o vencedor por regra.
- :mod:`~ficha.audit.confidence`     — regra declarada de ``confianca`` e fichas finais.
- :mod:`~ficha.audit.summary`        — tudo junto, em tabelas.
"""

from ficha.audit.confidence import (
    RULE_V1,
    RULE_V2,
    ConfidenceRule,
    FichaAuditada,
    build_final_fichas,
)
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
from ficha.audit.leakage import (
    LeakageResult,
    LeakageSummary,
    LeakedField,
    check_fewshot_leakage,
    leakage_summary,
)
from ficha.audit.normalize import normalize_for_match, split_context_pages
from ficha.audit.prompt_compare import PromptComparison, PromptVerdict, compare_prompt_variants
from ficha.audit.runstats import RunStats, run_stats
from ficha.audit.stability import StabilityReport, stability_report
from ficha.audit.summary import (
    AuditSummary,
    build_audit_summary,
    confidence_by_rule,
    confidence_distribution,
    fichas_nao_defensaveis,
)
from ficha.audit.temperature import TemperatureReport, temperature_report

__all__ = [
    "FIELDS",
    "RULE_V1",
    "RULE_V2",
    "AuditSummary",
    "ConfidenceRule",
    "DiffReport",
    "FichaAuditada",
    "FidelityResult",
    "FidelitySummary",
    "FieldDiff",
    "InputEffectReport",
    "LeakageResult",
    "LeakageSummary",
    "LeakedField",
    "LimitacaoCheck",
    "PromptComparison",
    "PromptVerdict",
    "RunStats",
    "StabilityReport",
    "TemperatureReport",
    "build_audit_summary",
    "build_final_fichas",
    "check_fewshot_leakage",
    "check_fidelity",
    "check_limitacao_support",
    "compare_prompt_variants",
    "confidence_by_rule",
    "confidence_distribution",
    "diff_runs",
    "fichas_nao_defensaveis",
    "fidelity_summary",
    "input_effect_report",
    "leakage_summary",
    "normalize_for_match",
    "run_stats",
    "split_context_pages",
    "stability_report",
    "temperature_report",
]
