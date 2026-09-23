"""Cache do texto limpo em JSON (``data/processed/<arquivo>.json``).

Dois motivos: (1) não reprocessar 476 páginas a cada execução do notebook; (2) rastreabilidade —
o JSON guarda o ``sha256`` do texto exato usado, que também vai para
``ExtractionRecord.doc_sha256``. Se o texto do cache não bater com o hash, o arquivo foi
alterado à mão e é rejeitado.

O cache é invalidado quando o PDF muda (hash dos bytes do PDF) ou quando a versão da limpeza
(:data:`CLEAN_VERSION`) muda.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ficha.ingest.pdf import load_pdf
from ficha.types import Document, Page

CLEAN_VERSION = 1
"""Incrementar sempre que ``ingest.clean`` mudar de comportamento, para invalidar o cache."""


def file_sha256(path: Path) -> str:
    """Hash dos bytes do arquivo (identifica a versão do PDF de origem)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cache_path(arquivo: str, processed_dir: Path) -> Path:
    """Caminho do JSON de cache para um PDF (``<processed_dir>/<arquivo>.json``)."""
    return Path(processed_dir) / f"{arquivo}.json"


def save_document(doc: Document, processed_dir: Path, *, source_sha256: str | None = None) -> Path:
    """Grava o documento em JSON e devolve o caminho. Cria o diretório se preciso."""
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "arquivo": doc.arquivo,
        "sha256": doc.sha256(),
        "source_sha256": source_sha256,
        "clean_version": CLEAN_VERSION,
        "metadata": doc.metadata,
        "pages": [{"number": p.number, "text": p.text} for p in doc.pages],
    }
    path = cache_path(doc.arquivo, processed_dir)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)
    return path


def _read_payload(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: cache inválido (esperado objeto JSON)")
    return data


def _payload_to_document(data: dict[str, Any], path: Path) -> Document:
    pages = tuple(Page(number=int(p["number"]), text=str(p["text"])) for p in data["pages"])
    doc = Document(arquivo=str(data["arquivo"]), pages=pages, metadata=dict(data["metadata"]))
    expected = data.get("sha256")
    if expected is not None and doc.sha256() != expected:
        raise ValueError(
            f"{path}: sha256 do texto não confere com o registrado — cache alterado ou corrompido."
        )
    return doc


def load_document(path: Path) -> Document:
    """Lê um documento do cache, verificando o ``sha256`` do texto."""
    path = Path(path)
    return _payload_to_document(_read_payload(path), path)


def load_or_ingest(pdf_path: Path, processed_dir: Path, *, force: bool = False) -> Document:
    """Devolve o documento do cache se ele for válido para este PDF; senão, ingere e grava.

    Válido = mesmo hash dos bytes do PDF, mesma :data:`CLEAN_VERSION` e hash do texto conferindo.
    """
    pdf_path = Path(pdf_path)
    src_hash = file_sha256(pdf_path)
    path = cache_path(pdf_path.name, processed_dir)
    if path.is_file() and not force:
        try:
            data = _read_payload(path)
            if data.get("source_sha256") == src_hash and data.get("clean_version") == CLEAN_VERSION:
                return _payload_to_document(data, path)
        except (ValueError, KeyError, TypeError):
            pass  # cache inválido: reprocessa abaixo
    doc = load_pdf(pdf_path)
    save_document(doc, processed_dir, source_sha256=src_hash)
    return doc
