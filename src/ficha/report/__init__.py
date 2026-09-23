"""Entregáveis da Seção 5: tabela CSV/XLSX, relatório PDF (≤ 3 páginas) e figuras.

- :mod:`ficha.report.table`   — ``fichas_to_frame`` / ``write_table`` (CSV com BOM + XLSX).
- :mod:`ficha.report.content` — ``ReportContent``: um campo por item exigido no relatório.
- :mod:`ficha.report.pdf`     — ``build_report_pdf`` com verificação rígida de páginas.
- :mod:`ficha.report.figures` — gráficos matplotlib (importado sob demanda: exige o extra
  ``notebook``).
- :mod:`ficha.report.naming`  — ``AtividadeI_<sobrenomes>``.
"""

from ficha.report.content import (
    AuditBlock,
    AuditResults,
    CostSection,
    PromptVersion,
    ReportContent,
    sample_content,
)
from ficha.report.naming import PLACEHOLDER_BASENAME, delivery_basename
from ficha.report.pdf import ReportTooLongError, build_report_pdf, count_pages
from ficha.report.table import TableOutputs, fichas_to_frame, write_table

__all__ = [
    "PLACEHOLDER_BASENAME",
    "AuditBlock",
    "AuditResults",
    "CostSection",
    "PromptVersion",
    "ReportContent",
    "ReportTooLongError",
    "TableOutputs",
    "build_report_pdf",
    "count_pages",
    "delivery_basename",
    "fichas_to_frame",
    "sample_content",
    "write_table",
]
