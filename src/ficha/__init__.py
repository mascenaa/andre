"""ficha — extração auditável de fichas comparativas de artigos científicos.

Pacote da Atividade de Construção I (ESPM · Computação Cognitiva · 2026.2).

Camadas (cada uma é um subpacote, com fronteira clara e testável isoladamente):

- ``ingest``   : PDF → :class:`ficha.types.Document` (texto limpo, página a página).
- ``select``   : Document → :class:`ficha.types.Context` (o que vai para o modelo — Seção 4.1).
- ``prompts``  : Context → prompt com técnicas identificáveis e ligáveis/desligáveis (Seção 4.2).
- ``llm``      : clientes de modelo (Qwen local, API, e um cliente falso determinístico).
- ``extract``  : orquestra seleção → prompt → modelo → parse/validação → persistência bruta.
- ``audit``    : fidelidade, estabilidade, efeito da entrada, temperatura, regra de confiança (4.4).
- ``cost``     : contagem de tokens e custo com premissa declarada (4.5).
- ``report``   : tabela CSV/XLSX e relatório PDF (Seção 5).
"""

from ficha.schema import Confianca, Evidencia, Ficha, FichaExtraida
from ficha.types import (
    Chunk,
    Completion,
    Context,
    Document,
    ExtractionRecord,
    GenerationParams,
    Page,
    ParseStatus,
    Usage,
)

__all__ = [
    "Chunk",
    "Completion",
    "Confianca",
    "Context",
    "Document",
    "Evidencia",
    "ExtractionRecord",
    "Ficha",
    "FichaExtraida",
    "GenerationParams",
    "Page",
    "ParseStatus",
    "Usage",
]

__version__ = "0.1.0"
