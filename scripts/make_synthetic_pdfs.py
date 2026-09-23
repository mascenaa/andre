"""Gera artigos científicos sintéticos em PDF (modo de ensaio e testes).

Uso:
    python scripts/make_synthetic_pdfs.py --out data/raw_synthetic --n 5 --seed 42

A implementação vive em ``ficha.ingest.synthetic`` (importável pelo notebook e pelos testes sem
mexer em ``sys.path``); este script é só a linha de comando. ``make_synthetic_corpus`` é
reexportado aqui para quem preferir importar do script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ficha.ingest.synthetic import MANIFEST_NAME, generate_synthetic_corpus, make_synthetic_corpus

__all__ = ["main", "make_synthetic_corpus"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/raw_synthetic"))
    parser.add_argument("--n", type=int, default=5, help="Quantidade de artigos (padrão 5).")
    parser.add_argument("--seed", type=int, default=42, help="Semente (padrão 42).")
    args = parser.parse_args(argv)
    articles = generate_synthetic_corpus(args.out, args.n, args.seed)
    for a in articles:
        lim = "com limitações" if a.has_limitations else "SEM limitações (esperado null)"
        n_hyph = len(a.hyphenated_words)
        print(f"{args.out / a.arquivo}  {a.n_pages} p.  {lim}  hifenizadas={n_hyph}")
    print(f"Gabarito: {args.out / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
