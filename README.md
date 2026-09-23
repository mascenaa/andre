# Atividade de Construção I — A ficha que você teria de defender

Pipeline que extrai, com um modelo de linguagem, uma ficha estruturada de cada um dos 19 artigos
da bibliografia **e prova, com números, quais fichas são confiáveis** — ESPM · Computação
Cognitiva · 2026.2 · Prof. André Insardi.

O produto não é a tabela: é a tabela **mais** a auditoria que diz de onde saiu cada linha
(fidelidade do trecho de evidência, estabilidade, efeito da entrada, efeito da temperatura), uma
regra declarada de `confianca` e a lista das fichas que o grupo **não** defenderia, com o motivo.

## Entregáveis (Seção 5)

| # | Arquivo | Gerado por |
|---|---|---|
| 1 | `AtividadeI_<sobrenomes>.ipynb` — executado, com saídas | `scripts/build_notebook.py` + "Reiniciar e executar tudo" |
| 2 | `AtividadeI_<sobrenomes>.csv` / `.xlsx` — 19 linhas, todos os campos | `ficha.report.write_table` (seção 11 do notebook) |
| 3 | `AtividadeI_<sobrenomes>.pdf` — ≤ 3 páginas | `ficha.report.build_report_pdf` (seção 12 do notebook) |

**Nomeação.** `ficha.report.delivery_basename(INTEGRANTES)` deriva `AtividadeI_<Sobrenome1>_<Sobrenome2>…`
(último sobrenome de cada integrante, sem acentos, na ordem declarada). Enquanto os integrantes
forem o placeholder, o nome é `AtividadeI_SOBRENOMES`. Para gerar o notebook já com os nomes e
o nome de arquivo certos:

```bash
python scripts/build_notebook.py --integrantes "Ana Souza" "João da Conceição"
# → notebooks/AtividadeI_Souza_Conceicao.ipynb  (a tabela e o PDF seguem a mesma lista)
```

## Estrutura do repositório

| Camada | Caminho | Responsabilidade | Seção do enunciado |
|---|---|---|---|
| Contratos | `src/ficha/{types,schema,config}.py` | tipos compartilhados, esquema da ficha (pydantic), `Settings` | 3, 7 |
| Ingestão | `src/ficha/ingest/` | PDF → texto limpo por página (hifenização, cabeçalhos, parágrafos), cache com `sha256` | 4.1 |
| Seleção | `src/ficha/select/` | `first_pages`, `keyword`, `semantic` (chunks + embeddings) | 4.1 |
| Prompt | `src/ficha/prompts/` | técnicas como blocos ligáveis; variantes `v_full`, `v_sem_fewshot`, ... | 4.2 |
| Modelo | `src/ficha/llm/` | Qwen2.5 local (T4), API (Anthropic / OpenAI-compatível), `FakeLLM` | 4.3 |
| Extração | `src/ficha/extract/` | parse + validação por esquema + reparos registrados; `RunStore` com saídas brutas | 4.2, 7 |
| Auditoria | `src/ficha/audit/` | fidelidade, estabilidade, entrada, temperatura, comparação de prompts, regra de confiança | 4.4, 4.2 |
| Custo | `src/ficha/cost/` | tokens reais × alternativa ingênua, premissa de preço visível | 4.5 |
| Entregáveis | `src/ficha/report/`, `src/ficha/cli.py` | tabela CSV/XLSX, relatório PDF, figuras, CLI | 5 |
| Notebook | `scripts/build_notebook.py` → `notebooks/` | orquestra e exibe; gerado por código | 5 |
| Decisões | `docs/adr/`, `docs/ARCHITECTURE.md` | por que cada escolha foi feita | todas |
| Roteiro do PDF | `docs/relatorio/RELATORIO_TEMPLATE.md` | esqueleto das 7 seções com marcadores `{{...}}` | 5 |

Dados (não versionados): `data/raw/` (os 19 PDFs), `data/processed/` (texto limpo),
`data/runs/<run_id>/` (`manifest.json` + `records.jsonl` com as **saídas brutas**),
`data/outputs/` (tabela, PDF, figuras).

## Mapa enunciado → código

| Enunciado | Onde |
|---|---|
| 4.1 Decidir o que mandar (e limpar o texto) | `ficha.ingest.clean`, `ficha.select` — ADR 0002 |
| 4.2 Técnicas de prompt + medição de duas versões | `ficha.prompts`, `ficha.audit.prompt_compare` — ADR 0003 |
| 4.2 JSON validado por esquema, malformado tratado | `ficha.schema.FichaExtraida`, `ficha.extract.parser` |
| 4.3 Modelo (Qwen2.5 na T4, ou API) e temperatura | `ficha.llm`, `ficha.config.Settings` — ADR 0004 |
| 4.4 a–d Auditoria + regra de `confianca` | `ficha.audit` — ADR 0005 |
| 4.5 Custo com premissa visível | `ficha.cost` |
| 5 Notebook, tabela, relatório ≤ 3 páginas | `scripts/build_notebook.py`, `ficha.report` — ADR 0006 |
| 7 Chaves só no ambiente; sementes; saídas brutas | `Settings` (`SecretStr`), `extract.set_seeds`, `RunStore` |

## Rodar localmente

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e Python 3.12.

```bash
make setup      # .venv + pacote em modo editável com extras dev/notebook/embeddings/api
make smoke      # pipeline inteiro em ~1 s: FakeLLM + PDFs sintéticos (sem GPU, sem rede)
make test       # pytest (inclui executar o notebook em modo ensaio)
make lint typecheck
make notebook   # gera e executa o notebook de entrega em modo real (MODO=ensaio só para testar)
make report     # relatório PDF de exemplo em data/outputs/
make entrega    # copia os três entregáveis para entrega/ (o que vai para o Canvas)
make help       # todos os alvos
```

`ficha smoke` imprime os run_ids, as quatro verificações com números, a comparação de prompts,
a distribuição de confiança, as fichas não defensáveis, o custo e o número de páginas do PDF.
A CLI tem também os passos isolados (`ficha ingest | extract | audit | cost | report`,
veja `ficha --help`).

**Modo ensaio × modo real.** O notebook lê `FICHA_MODO` (ou a variável `MODO` na célula de
setup). `ensaio` usa o `FakeLLM` com erros simulados (JSON quebrado, trecho inventado,
limitação inventada, instabilidade) sobre PDFs sintéticos — prova que o pipeline e a auditoria
funcionam, mas **seus números não são resultado**. `real` lê `data/raw` e usa o backend das
`Settings` (padrão do notebook: `qwen_local`).

## Rodar no Colab (entrega)

1. **Antes de qualquer célula:** *Ambiente de execução → Alterar tipo de ambiente → GPU T4*
   (a troca reinicia a sessão). O notebook já vem com `gpuType: T4` nos metadados.
2. Localmente, `make colab-zip` → `dist/ficha.zip` (código, sem dados).
3. No Colab, suba o notebook e `ficha.zip` para `/content/`. A célula de setup descompacta em
   `/content/ficha` e instala `ficha[local,embeddings,api]`.
4. Suba os 19 PDFs para `/content/ficha/data/raw/` (ou monte o Drive e aponte `RAW_DIR`).
5. Se usar API: cadastre `ANTHROPIC_API_KEY` nos **Secrets** do Colab (ícone de chave, com
   acesso liberado ao notebook). A célula de setup copia para o ambiente e só imprime se a chave
   existe — **nunca** o valor. Nada de chave no código ou no notebook.
6. Troque `MODO = "real"` (e, se for o caso, `BACKEND_REAL`), preencha `INTEGRANTES` e a
   primeira célula, e rode **Ambiente de execução → Reiniciar e executar tudo**.
7. Revise as conclusões do relatório em `Narrative(...)` (seção 12), regenere o PDF e confira
   que tem ≤ 3 páginas. Baixe o `.ipynb` executado, o CSV/XLSX e o PDF de
   `/content/ficha/data/outputs/`, todos com o nome `AtividadeI_<sobrenomes>`.

## Decisões (ADRs)

- [0001 — Pacote testável em vez de notebook monolítico](docs/adr/0001-pacote-testavel-em-vez-de-notebook-monolitico.md)
- [0002 — Seleção de contexto](docs/adr/0002-selecao-de-contexto.md)
- [0003 — Engenharia de prompt](docs/adr/0003-engenharia-de-prompt.md)
- [0004 — Escolha do modelo](docs/adr/0004-escolha-do-modelo.md)
- [0005 — Auditoria e regra de confiança](docs/adr/0005-auditoria-e-regra-de-confianca.md)
- [0006 — Entregáveis: notebook gerado, tabela com BOM, PDF com limite verificado](docs/adr/0006-entregaveis.md)

Contratos entre camadas: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Enunciado integral:
[`docs/ENUNCIADO.md`](docs/ENUNCIADO.md).

## Regras seguidas (Seção 7)

- **Sem framework de extração ou RAG** (nada de LangChain, LlamaIndex, instructor, outlines):
  prompt, validação e auditoria são código próprio; `pymupdf`, `sentence-transformers` e
  `rapidfuzz` são bibliotecas permitidas.
- **Chaves só por ambiente** (`.env` ignorado pelo git, ou Secrets do Colab), lidas como
  `SecretStr` e nunca impressas. Um teste executa o notebook com uma chave-sentinela no
  ambiente e falha se ela aparecer em qualquer saída.
- **Reprodutibilidade:** sementes fixas (`random`, `numpy`, `torch`), temperatura e semente no
  `manifest.json` de cada execução, saídas brutas preservadas em `data/runs/`.

## Declaração de uso de IA

Usamos assistentes de IA (Claude) para gerar partes do código, dos testes e da documentação
deste repositório, e para revisar textos. As decisões de projeto estão registradas nos ADRs; o
grupo revisou o código, roda os testes e sabe explicar cada linha entregue. Os números do
relatório vêm exclusivamente das execuções registradas em `data/runs/`.
