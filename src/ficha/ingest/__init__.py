"""Ingestão: PDF → :class:`ficha.types.Document` com texto limpo, página a página.

- :mod:`ficha.ingest.pdf`       — leitura com ``pymupdf`` (``load_pdf``, ``load_corpus``).
- :mod:`ficha.ingest.clean`     — limpeza determinística e conservadora (``clean_text``).
- :mod:`ficha.ingest.cache`     — cache JSON com ``sha256`` (``load_or_ingest``).
- :mod:`ficha.ingest.synthetic` — artigos sintéticos em PDF para testes e modo de ensaio.
"""

from ficha.ingest.cache import load_document, load_or_ingest, save_document
from ficha.ingest.clean import clean_text, strip_repeated_lines
from ficha.ingest.pdf import EmptyDocumentError, PdfLoadError, load_corpus, load_pdf

__all__ = [
    "EmptyDocumentError",
    "PdfLoadError",
    "clean_text",
    "load_corpus",
    "load_document",
    "load_or_ingest",
    "load_pdf",
    "save_document",
    "strip_repeated_lines",
]
