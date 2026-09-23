"""Notebook de entrega: gerado e executado de ponta a ponta em modo ensaio (Seção 5)."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import nbformat
import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
SECRET = "sk-teste-NUNCA-deve-aparecer-0123456789"


def _builder():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(
        "build_notebook", ROOT / "scripts/build_notebook.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_notebook"] = module
    spec.loader.exec_module(module)
    return module


def test_builder_structure_and_naming(tmp_path: Path) -> None:
    build = _builder()
    path = build.build([], tmp_path / "nb.ipynb")
    nb = nbformat.read(path, as_version=4)
    nbformat.validate(nb)
    first = nb.cells[0]
    assert first.cell_type == "markdown"
    assert "Integrantes" in first.source and "<<INTEGRANTES" in first.source
    assert "uso de IA" in first.source
    assert nb.metadata["colab"]["gpuType"] == "T4"
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    for needed in (
        "ExtractionRunner",
        "build_audit_summary",
        "write_table",
        "build_report_pdf",
        "naive_cost_from_records",
        "describe_diff",
        "userdata.get",
    ):
        assert needed in code, needed
    assert "print(valor" not in code and "print(os.environ[" not in code

    named = build.build(["Ana Souza", "João da Conceição"], None)
    try:
        assert named.name == "AtividadeI_Souza_Conceicao.ipynb"
        assert "Ana Souza" in nbformat.read(named, as_version=4).cells[0].source
    finally:
        named.unlink()


def test_notebook_runs_end_to_end_in_rehearsal_mode(tmp_path: Path, monkeypatch) -> None:
    from nbclient import NotebookClient

    monkeypatch.setenv("FICHA_MODO", "ensaio")
    monkeypatch.setenv("FICHA_ROOT", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    build = _builder()
    path = build.build([], tmp_path / "AtividadeI_SOBRENOMES.ipynb")
    nb = nbformat.read(path, as_version=4)

    t0 = time.perf_counter()
    NotebookClient(
        nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(tmp_path)}}
    ).execute()
    elapsed = time.perf_counter() - t0
    assert elapsed < 180, f"notebook levou {elapsed:.0f} s"

    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    assert all(c.get("execution_count") for c in code_cells)
    assert not any(o.get("output_type") == "error" for c in code_cells for o in c.outputs)
    assert sum(bool(c.outputs) for c in code_cells) == len(code_cells), "toda célula tem saída"
    assert SECRET not in json.dumps(nb), "a chave nunca pode aparecer no notebook"

    out = tmp_path / "data" / "outputs" / "ensaio" / "outputs"
    assert (out / "AtividadeI_SOBRENOMES.csv").exists()
    assert (out / "AtividadeI_SOBRENOMES.xlsx").exists()
    pdf = out / "AtividadeI_SOBRENOMES.pdf"
    with pymupdf.open(pdf) as doc:
        assert 1 <= doc.page_count <= 3
    runs = [d for d in (tmp_path / "data" / "outputs" / "ensaio" / "runs").iterdir() if d.is_dir()]
    # 5 da matriz (estratégia principal hybrid) + a execução semantic de "antes" (seção 8c)
    assert len(runs) == 6
    assert any("_semantic_v_full_t0.0_rep1" in d.name for d in runs)


@pytest.mark.skipif(not (ROOT / "notebooks").exists(), reason="sem diretório de notebooks")
def test_committed_notebook_is_current() -> None:
    """O .ipynb versionado tem a mesma estrutura que o gerador produz (não foi editado à mão)."""
    committed = ROOT / "notebooks" / "AtividadeI_SOBRENOMES.ipynb"
    if not committed.exists():
        pytest.skip("notebook ainda não gerado")
    fresh = [c.source for c in _builder().cells([])]
    current = [c.source for c in nbformat.read(committed, as_version=4).cells]
    assert current == fresh, "rode `make notebook` para regenerar o .ipynb"
