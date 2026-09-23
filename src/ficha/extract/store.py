"""Persistência das execuções: ``<runs_dir>/<run_id>/{manifest.json, records.jsonl}``.

Uma linha de ``records.jsonl`` por chamada ao modelo (:class:`ExtractionRecord`), com a saída
**bruta** — sem ela não há verificação de estabilidade (Seção 7, Reprodutibilidade). O
``manifest.json`` declara tudo o que o relatório precisa (modelo, variante, estratégia,
temperatura, semente, totais de tokens).

Formato do ``run_id`` (ver ``docs/ARCHITECTURE.md``):
``<YYYYmmdd-HHMMSS>_<strategy>_<variant>_t<temp>_rep<n>``. Se dois ids colidirem no mesmo
segundo, acrescenta-se ``_2``, ``_3``...
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from ficha.types import ExtractionRecord

RECORDS_FILE = "records.jsonl"
MANIFEST_FILE = "manifest.json"

_UNSAFE = re.compile(r"[^A-Za-z0-9._\-]+")


def _slug(value: str) -> str:
    """Parte segura para nome de diretório.

    Underscores são mantidos (``first_pages``, ``v_full`` aparecem literalmente no id), então o
    id serve para leitura humana; para obter os campos, use o manifest, não o nome.
    """
    return _UNSAFE.sub("-", value).strip("-") or "x"


def format_temperature(temperature: float) -> str:
    """``0.0`` → ``"0.0"``, ``0.7`` → ``"0.7"``, ``0.25`` → ``"0.25"``."""
    text = f"{temperature:.2f}".rstrip("0")
    return text + "0" if text.endswith(".") else text


class RunStore:
    """Lê e grava execuções em disco. Não guarda estado além do diretório."""

    def __init__(self, runs_dir: Path | str) -> None:
        self.runs_dir = Path(runs_dir)

    # ------------------------------------------------------------------ ids e caminhos
    def new_run_id(
        self,
        strategy: str,
        variant: str,
        temperature: float,
        repetition: int = 1,
        now: datetime | None = None,
    ) -> str:
        """Gera um ``run_id`` novo (e reserva o diretório, para não colidir)."""
        stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
        base = (
            f"{stamp}_{_slug(strategy)}_{_slug(variant)}"
            f"_t{format_temperature(temperature)}_rep{repetition}"
        )
        run_id, k = base, 2
        while self.run_dir(run_id).exists():
            run_id, k = f"{base}_{k}", k + 1
        self.run_dir(run_id).mkdir(parents=True, exist_ok=False)
        return run_id

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def records_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / RECORDS_FILE

    def manifest_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / MANIFEST_FILE

    # ------------------------------------------------------------------ escrita
    def append(self, run_id: str, record: ExtractionRecord) -> None:
        """Acrescenta um registro (uma linha JSON) — gravado a cada artigo, não no fim."""
        path = self.records_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def write_manifest(self, run_id: str, manifest: dict[str, Any]) -> None:
        """Grava (sobrescreve) o manifest da execução."""
        path = self.manifest_path(run_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    # ------------------------------------------------------------------ leitura
    def load(self, run_id: str) -> list[ExtractionRecord]:
        """Todos os registros da execução, na ordem de gravação."""
        path = self.records_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"Execução sem registros: {path}")
        with path.open(encoding="utf-8") as fh:
            return [ExtractionRecord.from_dict(json.loads(line)) for line in fh if line.strip()]

    def manifest(self, run_id: str) -> dict[str, Any]:
        """Manifest da execução."""
        path = self.manifest_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"Execução sem manifest: {path}")
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def list_runs(self) -> list[str]:
        """``run_id`` de todas as execuções com registros ou manifest, em ordem cronológica."""
        if not self.runs_dir.exists():
            return []
        return sorted(
            d.name
            for d in self.runs_dir.iterdir()
            if d.is_dir() and ((d / RECORDS_FILE).exists() or (d / MANIFEST_FILE).exists())
        )

    def __repr__(self) -> str:
        return f"RunStore(runs_dir={str(self.runs_dir)!r})"
