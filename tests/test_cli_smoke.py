from __future__ import annotations

import re
from pathlib import Path

import pymupdf
from typer.testing import CliRunner

from ficha.cli import app, run_rehearsal
from ficha.report.pdf import SECTION_TITLES

runner = CliRunner()


def test_smoke_command_runs_end_to_end(tmp_path: Path) -> None:
    result = runner.invoke(app, ["smoke", "--n-docs", "4", "--keep", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "a) Fidelidade" in result.output
    assert "d) Efeito da temperatura" in result.output
    assert "Custo" in result.output
    outputs = tmp_path / "outputs"
    assert (outputs / "AtividadeI_SOBRENOMES.csv").exists()
    assert (outputs / "AtividadeI_SOBRENOMES.xlsx").exists()
    assert (outputs / "AtividadeI_SOBRENOMES.pdf").exists()
    # saídas brutas preservadas: 5 execuções com manifest + records
    runs = [d for d in (tmp_path / "runs").iterdir() if d.is_dir()]
    assert len(runs) == 5
    assert all((d / "records.jsonl").exists() and (d / "manifest.json").exists() for d in runs)


def test_rehearsal_result_is_consistent(tmp_path: Path) -> None:
    res = run_rehearsal(tmp_path, n_docs=4, integrantes=["Ana Souza", "João da Conceição"])
    assert set(res.run_ids) == {"primary", "stability", "prompt_alt", "input_alt", "temp_alt"}
    assert len(res.summary.fichas) == 4
    assert res.pdf.name == "AtividadeI_Souza_Conceicao.pdf"
    assert 1 <= res.pdf_pages <= 3
    with pymupdf.open(res.pdf) as pdf:
        text = "\n".join(p.get_text() for p in pdf)
    for title in SECTION_TITLES:
        assert title in text
    assert "Ana Souza" in text
    assert res.cost.real.n_calls == 4 * 4 + 2  # 4 execuções completas + subconjunto t_alt


def test_report_exemplo(tmp_path: Path) -> None:
    out = tmp_path / "ex.pdf"
    result = runner.invoke(app, ["report", "--exemplo", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_report_without_exemplo_points_to_notebook() -> None:
    result = runner.invoke(app, ["report"])
    assert result.exit_code == 1
    assert "notebook" in result.output


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("ingest", "extract", "audit", "cost", "report", "smoke"):
        assert cmd in result.output


def test_extract_audit_cost_commands(tmp_path: Path, monkeypatch) -> None:
    from ficha.ingest.synthetic import make_synthetic_corpus

    monkeypatch.setenv("FICHA_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FICHA_MODEL_BACKEND", "fake")
    make_synthetic_corpus(tmp_path / "raw", n=3, seed=7)

    assert runner.invoke(app, ["ingest"]).exit_code == 0
    ids: dict[str, str] = {}
    plan = {
        "primary": ["--strategy", "semantic", "--variant", "v_full", "--repetition", "1"],
        "stability": ["--strategy", "semantic", "--variant", "v_full", "--repetition", "2"],
        "input-alt": ["--strategy", "first_pages", "--variant", "v_full"],
        "temp-alt": ["--strategy", "semantic", "--temperature", "0.7", "--limit", "2"],
        "prompt-alt": ["--strategy", "semantic", "--variant", "v_sem_fewshot"],
    }
    for key, args in plan.items():
        res = runner.invoke(app, ["extract", *args, "--hashing-embedder"])
        assert res.exit_code == 0, res.output
        match = re.search(r"run_id: (\S+)", res.output)
        assert match, res.output
        ids[key] = match.group(1)

    audit_args = ["audit"]
    for key, rid in ids.items():
        audit_args += [f"--{key}", rid]
    res = runner.invoke(app, audit_args)
    assert res.exit_code == 0, res.output
    assert (tmp_path / "outputs" / "audit_summary.json").exists()

    res = runner.invoke(app, ["cost"])
    assert res.exit_code == 0, res.output
    assert "Premissa" in res.output
