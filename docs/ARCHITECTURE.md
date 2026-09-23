# Arquitetura — `ficha`

> Este documento é o **contrato** entre as camadas. Quem implementa uma camada codifica
> contra os tipos de `src/ficha/types.py` e `src/ficha/schema.py`, nunca contra a
> implementação interna de outra camada.

## Por que um pacote, e não um notebook gigante

O enunciado pede um notebook executado de ponta a ponta. Mas a *lógica* — limpeza,
seleção, prompt, parse, auditoria, custo — vive num pacote Python testável (`src/ficha`),
e o notebook apenas **orquestra e exibe**. Isso dá:

- testes unitários para cada decisão (sem GPU, sem rede, sem os PDFs reais);
- decisões rastreáveis (cada estratégia/técnica é uma classe ou flag nomeada);
- o mesmo código roda no Colab (Qwen local) e localmente (cliente falso / API).

## Fluxo de dados

```
PDF ──ingest──▶ Document ──select──▶ Context ──prompts──▶ RenderedPrompt
                                                              │
                                              llm.generate ◀──┘
                                                  │
                         Completion ──extract.parser──▶ FichaExtraida | erro
                                                  │
                    ExtractionRecord (persistido em data/runs/<run_id>/records.jsonl)
                                                  │
        audit.{fidelity,stability,input_effect,temperature,prompt_compare}
                                                  │
                         audit.confidence ──▶ Ficha (com confianca) ──▶ report.{table,pdf}
                                                  │
                                    cost.calculator (a partir de Usage)
```

## Contratos (resumo — a fonte da verdade é `types.py`)

| Tipo | Produzido por | Consumido por |
|---|---|---|
| `Document(arquivo, pages)` | `ingest` | `select`, `audit` |
| `Context(arquivo, strategy, chunks).render()` | `select` | `prompts`, `extract` |
| `RenderedPrompt(variant, features, system, user)` | `prompts` | `extract`, `llm` |
| `Completion(text, usage, ...)` | `llm` | `extract` |
| `ExtractionRecord` (jsonl) | `extract` | `audit`, `cost`, `report` |
| `FichaExtraida` / `Ficha` | `extract` / `audit.confidence` | `report` |

### Regras invariantes

1. **`Context.render()` é o texto exato enviado.** Ele começa cada chunk com `[p. N]`.
   O `ExtractionRecord.context_text` guarda esse texto. A fidelidade (4.4a) compara
   `evidencia.trecho` contra **ele**, não contra o PDF inteiro.
2. **O modelo nunca preenche `confianca`.** `FichaExtraida` não tem esse campo; ele é
   atribuído por `audit.confidence` segundo regra declarada.
3. **Saída bruta é sagrada.** `Completion.text` vai para `raw_output` sem alteração.
   Reparos acontecem no parser e são marcados como `ParseStatus.REPAIRED`.
4. **Delimitadores fixos.** O prompt envolve o artigo em `<<<ARTIGO>>> ... <<<FIM_ARTIGO>>>`.
   O `FakeLLM` depende disso para localizar o texto.
5. **Sem framework de extração/RAG.** `sentence-transformers` para embeddings é permitido;
   LangChain/LlamaIndex/instructor/outlines não.
6. **Chaves só por ambiente** (`Settings`, `SecretStr`). Nunca logar.
7. **Seeds fixas** em tudo que tiver aleatoriedade (torch, numpy, random).

## Layout e ownership (paralelização)

| Área | Caminhos | Dono |
|---|---|---|
| Fundação (contratos) | `src/ficha/{__init__,types,schema,config}.py`, `tests/conftest.py`, `pyproject.toml` | lead |
| Ingestão + Seleção | `src/ficha/ingest/**`, `src/ficha/select/**`, `scripts/make_synthetic_pdfs.py`, `tests/test_ingest*.py`, `tests/test_select*.py`, `docs/adr/0002-*.md` | implementer-1 |
| Modelo + Prompt + Extração | `src/ficha/llm/**`, `src/ficha/prompts/**`, `src/ficha/extract/**`, `tests/test_llm*.py`, `tests/test_prompts*.py`, `tests/test_extract*.py`, `docs/adr/0003-*.md`, `docs/adr/0004-*.md` | implementer-2 |
| Auditoria + Custo | `src/ficha/audit/**`, `src/ficha/cost/**`, `tests/test_audit*.py`, `tests/test_cost*.py`, `docs/adr/0005-*.md` | implementer-3 |
| Relatório + Notebook + CLI + Docs | `src/ficha/report/**`, `src/ficha/cli.py`, `notebooks/**`, `docs/relatorio/**`, `README.md`, `Makefile`, `docs/adr/0006-*.md` | implementer-4 |

## Convenções

- Python 3.12, `ruff` (line-length 100, docstrings obrigatórias em `src`), `mypy --strict`.
- Docstrings e comentários em **português**; identificadores em inglês, exceto os nomes
  de campos da ficha (`problema`, `dados`, ...) que seguem o enunciado à risca.
- Cada decisão de projeto não óbvia vira um ADR em `docs/adr/` (formato: Contexto,
  Decisão, Alternativas, Consequências).
- Testes: `pytest`, sem rede. Testes que baixam modelo levam `@pytest.mark.slow`.
- Persistência de execuções: `data/runs/<run_id>/manifest.json` + `records.jsonl`.
  `run_id = <YYYYmmdd-HHMMSS>_<strategy>_<variant>_t<temp>_rep<n>`.
