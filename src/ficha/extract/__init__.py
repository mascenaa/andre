"""Extração: seleção → prompt → modelo → parse/validação → persistência bruta.

- :mod:`ficha.extract.parser` — :func:`parse_completion` (validação e reparos registrados).
- :mod:`ficha.extract.store`  — :class:`RunStore` (``manifest.json`` + ``records.jsonl``).
- :mod:`ficha.extract.runner` — :class:`ExtractionRunner` e :class:`RunResult`.
"""

from ficha.extract.parser import ParseResult, parse_completion
from ficha.extract.runner import ExtractionRunner, RunResult, reparse, set_seeds
from ficha.extract.store import RunStore

__all__ = [
    "ExtractionRunner",
    "ParseResult",
    "RunResult",
    "RunStore",
    "parse_completion",
    "reparse",
    "set_seeds",
]
