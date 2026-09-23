"""Leitura de PDF → :class:`ficha.types.Document`, página a página, com texto limpo.

Usa ``pymupdf`` (biblioteca de leitura de PDF, permitida pela Seção 7). O texto de cada página
é extraído por **blocos** na ordem do fluxo de conteúdo do PDF — que, em artigos de duas
colunas gerados por LaTeX, respeita a ordem de leitura (coluna esquerda, depois direita).
Ordenar por coordenada (``sort=True``) intercalaria as colunas linha a linha, então não fazemos.
Os blocos são unidos por linha em branco, o que dá a :func:`ficha.ingest.clean.clean_text` a
fronteira de parágrafo que o PDF conhece.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pymupdf

from ficha.ingest.clean import build_vocabulary, clean_text, strip_repeated_lines
from ficha.types import Document, Page

logger = logging.getLogger(__name__)

MIN_CHARS_PER_PAGE = 20
"""Abaixo desta média de caracteres limpos por página, o PDF é tratado como sem texto."""


class PdfLoadError(RuntimeError):
    """O arquivo não pôde ser aberto como PDF."""


class EmptyDocumentError(ValueError):
    """O PDF abriu, mas não tem camada de texto utilizável (provavelmente escaneado)."""


def extract_raw_pages(path: Path) -> tuple[list[str], dict[str, Any]]:
    """Extrai o texto bruto de cada página e os metadados do PDF.

    Blocos de texto unidos por linha em branco; nada é limpo aqui — ver :func:`load_pdf`.
    """
    try:
        pdf = pymupdf.open(path)  # type: ignore[no-untyped-call]
    except Exception as exc:  # pymupdf levanta tipos variados conforme a corrupção
        raise PdfLoadError(f"Não foi possível abrir {path} como PDF: {exc}") from exc
    with pdf:
        pages: list[str] = []
        for page in pdf:  # type: ignore[attr-defined]
            blocks = page.get_text("blocks")
            texts = [str(b[4]).strip("\n") for b in blocks if b[6] == 0 and str(b[4]).strip()]
            pages.append("\n\n".join(texts))
        meta: dict[str, Any] = dict(pdf.metadata or {})
    return pages, meta


def load_pdf(path: Path) -> Document:
    """Carrega um PDF como :class:`Document`: uma :class:`Page` por página (1-based).

    Etapas: extração por blocos → remoção de cabeçalho/rodapé repetido
    (:func:`strip_repeated_lines`) → :func:`clean_text` em cada página, com o vocabulário do
    documento inteiro como evidência para a hifenização.

    Raises:
        FileNotFoundError: o arquivo não existe.
        PdfLoadError: o arquivo não é um PDF legível.
        EmptyDocumentError: o PDF não tem texto extraível (ex.: digitalizado sem OCR).

    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF não encontrado: {path}")
    raw_pages, pdf_meta = extract_raw_pages(path)
    n_chars_raw = sum(len(p) for p in raw_pages)

    stripped = strip_repeated_lines(raw_pages)
    vocabulary = build_vocabulary(stripped)
    cleaned = [clean_text(p, vocabulary) for p in stripped]
    n_chars_clean = sum(len(p) for p in cleaned)

    n_pages = len(raw_pages)
    if n_pages == 0 or n_chars_clean < MIN_CHARS_PER_PAGE * n_pages:
        raise EmptyDocumentError(
            f"{path.name}: {n_pages} página(s) mas só {n_chars_clean} caracteres de texto. "
            "O PDF provavelmente é digitalizado (imagem, sem camada de texto). "
            "Rode OCR antes (ex.: `ocrmypdf entrada.pdf saida.pdf`) ou remova-o do corpus."
        )

    empty_pages = [i + 1 for i, t in enumerate(cleaned) if not t]
    if empty_pages:
        logger.warning("%s: páginas sem texto: %s", path.name, empty_pages)

    title = (pdf_meta.get("title") or "").strip() or None
    metadata: dict[str, Any] = {
        "source_path": str(path),
        "n_pages": n_pages,
        "n_chars_raw": n_chars_raw,
        "n_chars_clean": n_chars_clean,
        "title": title,
        "author": (pdf_meta.get("author") or "").strip() or None,
        "empty_pages": empty_pages,
    }
    pages = tuple(Page(number=i + 1, text=t) for i, t in enumerate(cleaned))
    return Document(arquivo=path.name, pages=pages, metadata=metadata)


def load_corpus(
    directory: Path, pattern: str = "*.pdf", *, skip_errors: bool = False
) -> list[Document]:
    """Carrega todos os PDFs de ``directory`` que casam com ``pattern``, ordenados por nome.

    Com ``skip_errors=True``, PDFs ilegíveis ou sem texto são registrados em log e pulados
    (útil para não perder a execução inteira por um arquivo); o padrão é falhar alto.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Diretório do corpus não encontrado: {directory}")
    docs: list[Document] = []
    for path in sorted(directory.glob(pattern), key=lambda p: p.name):
        try:
            docs.append(load_pdf(path))
        except (PdfLoadError, EmptyDocumentError) as exc:
            if not skip_errors:
                raise
            logger.warning("Pulando %s: %s", path.name, exc)
    return docs
