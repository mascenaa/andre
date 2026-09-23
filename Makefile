# Atalhos do projeto. O caminho do repositório tem espaço ("Atividade Andre"):
# por isso todo caminho é derivado de $(CURDIR) e sempre usado entre aspas.

SHELL      := /bin/bash
.SHELLFLAGS := -ec
.ONESHELL:
.DEFAULT_GOAL := help

ROOT       := $(CURDIR)
VENV       := $(ROOT)/.venv
PY         := "$(VENV)/bin/python"
NOTEBOOK   := notebooks/AtividadeI_SOBRENOMES.ipynb
EXTRAS     := dev,notebook,embeddings,api
# Modo do notebook/relatório: ensaio (FakeLLM + PDFs sintéticos) ou real (data/raw + Settings).
MODO       ?= ensaio

.PHONY: help setup lint format typecheck test test-all smoke notebook report colab-zip clean

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

notebook: ## Gera o .ipynb com nbformat e executa de ponta a ponta (saídas no próprio arquivo)
	cd "$(ROOT)"
	$(PY) scripts/build_notebook.py
	FICHA_MODO=$(MODO) $(PY) -m jupyter nbconvert --execute --to notebook --inplace \
		--ExecutePreprocessor.timeout=1800 "$(NOTEBOOK)"

report: ## Relatório PDF de exemplo (≤ 3 páginas) em data/outputs
	cd "$(ROOT)" && "$(VENV)/bin/ficha" report --exemplo

colab-zip: ## Empacota o código (sem dados) em dist/ficha.zip para subir no Colab
	cd "$(ROOT)"
	mkdir -p dist
	rm -f dist/ficha.zip
	zip -qr dist/ficha.zip pyproject.toml README.md src scripts notebooks docs \
		-x '*/__pycache__/*' '*.pyc' '*/.ipynb_checkpoints/*'
	@echo "dist/ficha.zip pronto"

clean: ## Remove caches e o cache de texto (preserva data/raw, data/runs e data/outputs)
	cd "$(ROOT)"
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build htmlcov .coverage
	find src tests scripts -name '__pycache__' -type d -prune -exec rm -rf {} +
	# data/runs guarda as saídas brutas do modelo (Seção 7): nunca apagar automaticamente.
	find data/processed -mindepth 1 ! -name '.gitkeep' -exec rm -rf {} +
