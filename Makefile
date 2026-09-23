# Atalhos do projeto. O caminho do repositório tem espaço ("Atividade Andre"):
# por isso todo caminho é derivado de $(CURDIR) e sempre usado entre aspas.

SHELL      := /bin/bash
.SHELLFLAGS := -ec
.ONESHELL:
.DEFAULT_GOAL := help

ROOT       := $(CURDIR)
VENV       := $(ROOT)/.venv
PY         := "$(VENV)/bin/python"
INTEGRANTES := "João Pedro Mascena" "Felipe Lira" "Felipe Murakami" "Pietro Garbin"
NOTEBOOK   := notebooks/AtividadeI_Mascena_Lira_Murakami_Garbin.ipynb
EXTRAS     := dev,notebook,embeddings,api
# Modo do notebook/relatório: real (data/raw + data/runs, a entrega) ou ensaio (FakeLLM +
# PDFs sintéticos, só para testar o pipeline: `make notebook MODO=ensaio` grava em
# data/outputs/ensaio e NÃO deve ser entregue).
MODO       ?= real

.PHONY: help setup lint format typecheck test test-all smoke notebook report colab-zip entrega clean

help: ## Lista os alvos
	@grep -E '^[a-zA-Z_-]+:.*?## ' "$(ROOT)/Makefile" | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Cria .venv (uv, Python 3.12) e instala o pacote com extras dev/notebook/embeddings
	cd "$(ROOT)"
	uv venv --python 3.12 "$(VENV)"
	uv pip install --python "$(VENV)/bin/python" -e ".[$(EXTRAS)]"

lint: ## ruff check
	cd "$(ROOT)" && $(PY) -m ruff check src tests scripts

format: ## ruff format + correções automáticas
	cd "$(ROOT)" && $(PY) -m ruff format src tests scripts && $(PY) -m ruff check --fix src tests scripts

typecheck: ## mypy --strict em src
	cd "$(ROOT)" && $(PY) -m mypy src

test: ## pytest rápido (sem os testes marcados como slow)
	cd "$(ROOT)" && $(PY) -m pytest -q -m "not slow"

test-all: ## pytest completo, incluindo o notebook em modo ensaio
	cd "$(ROOT)" && $(PY) -m pytest -q

smoke: ## Pipeline inteiro em 30 s: FakeLLM + PDFs sintéticos num diretório temporário
	cd "$(ROOT)" && "$(VENV)/bin/ficha" smoke

notebook: ## Gera o .ipynb de entrega (com os integrantes) e executa de ponta a ponta
	cd "$(ROOT)"
	$(PY) scripts/build_notebook.py --integrantes $(INTEGRANTES)
	FICHA_MODO=$(MODO) $(PY) -m jupyter nbconvert --execute --to notebook --inplace \
		--ExecutePreprocessor.timeout=1800 "$(NOTEBOOK)"

report: ## Relatório PDF de exemplo (≤ 3 páginas) em data/outputs
	cd "$(ROOT)" && "$(VENV)/bin/ficha" report --exemplo

entrega: ## Copia os três entregáveis (ipynb executado, csv/xlsx, pdf) para entrega/
	cd "$(ROOT)"
	mkdir -p entrega
	rm -f entrega/AtividadeI_*
	cp "$(NOTEBOOK)" entrega/
	cp data/outputs/AtividadeI_*.csv data/outputs/AtividadeI_*.xlsx data/outputs/AtividadeI_*.pdf entrega/
	@ls -la entrega

colab-zip: ## Empacota código + saídas brutas (data/runs) em dist/ficha.zip para o Colab
	cd "$(ROOT)"
	mkdir -p dist
	rm -f dist/ficha.zip
	# data/runs entra de propósito: são as saídas brutas (Seção 7) que o notebook reaproveita
	# para rodar de ponta a ponta sem refazer ~1 h de GPU. Os PDFs (data/raw) não entram.
	zip -qr dist/ficha.zip pyproject.toml README.md src scripts notebooks docs data/runs \
		-x '*/__pycache__/*' '*.pyc' '*/.ipynb_checkpoints/*'
	@echo "dist/ficha.zip pronto"

clean: ## Remove caches e o cache de texto (preserva data/raw, data/runs e data/outputs)
	cd "$(ROOT)"
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build htmlcov .coverage
	find src tests scripts -name '__pycache__' -type d -prune -exec rm -rf {} +
	# data/runs guarda as saídas brutas do modelo (Seção 7): nunca apagar automaticamente.
	find data/processed -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +
