# ADR 0006 — Entregáveis: notebook gerado, tabela com BOM, PDF com limite verificado

**Status:** aceito · **Data:** 2026-09-23

## Contexto

A Seção 5 pede três arquivos, nomeados `AtividadeI_<sobrenomes>`:

1. o notebook `.ipynb` **executado, com saídas visíveis**, que roda de ponta a ponta
   ("Reiniciar e executar tudo"); sem isso o trabalho não é corrigido;
2. a tabela dos 19 artigos em CSV ou XLSX, uma linha por artigo, com todos os campos da ficha;
3. um relatório PDF de **no máximo 3 páginas**, com sete itens obrigatórios e os integrantes
   identificados na primeira página (e na primeira célula do notebook).

Os três precisam contar a mesma história com os mesmos números, e o grupo precisa conseguir
regerá-los rapidamente quando uma execução real (Colab/T4) mudar os resultados.

## Decisão

### Notebook gerado por código (`scripts/build_notebook.py` → `nbformat`)

O `.ipynb` não é editado à mão: é **gerado** por um script Python que declara célula a célula
(markdown de narrativa + código que só orquestra o pacote). Motivos:

- **Diffável e revisável**: o script é Python legível em PR; o JSON do `.ipynb` com saídas não é.
- **Reprodutível**: regerar o notebook e executá-lo (`make notebook`) sempre produz a mesma
  estrutura; não há células "fantasmas" nem ordem de execução quebrada.
- **Testável**: `tests/test_notebook_smoke.py` gera e executa o notebook com `nbclient` em modo
  ensaio — é o teste de aceite do "roda de ponta a ponta".
- **Sem lógica no notebook** (ADR 0001): as células chamam `ficha.*`; se uma célula ficasse longa,
  isso indicaria lógica que deveria estar no pacote.

Dois modos, escolhidos por `MODO` na célula de setup (ou `FICHA_MODO` no ambiente):
`ensaio` (FakeLLM + PDFs sintéticos gerados na hora, sem GPU/rede, < 3 min) e `real`
(os 19 PDFs em `data/raw` + backend declarado nas `Settings`, no Colab com T4).

### Tabela: CSV UTF-8 com BOM + XLSX formatado

- Colunas na ordem exata da Seção 3 (`Ficha.to_row`); colunas de auditoria (motivos da
  confiança) vêm **depois**, nunca misturadas.
- CSV com BOM (`utf-8-sig`) porque o Excel pt-BR interpreta CSV sem BOM como Latin-1 e
  estraga os acentos. `limitacao = null` vira célula vazia, nunca o texto "None".
- XLSX com cabeçalho em negrito, quebra de linha, larguras fixas e painel congelado: é o
  arquivo que o coordenador fictício abriria.

### Relatório: `ReportContent` + `reportlab` + verificação de páginas

- `ReportContent` tem **um campo por item exigido** na Seção 5; esquecer um item é erro de
  construção, não uma seção que some.
- `reportlab` (platypus) em vez de exportar o notebook para PDF: controle fino de tipografia
  (8 pt, tabelas compactas) é o que torna 3 páginas viáveis com todos os números.
- Fonte DejaVu Sans (vem com o matplotlib) para acentos e símbolos; fallback Helvetica
  (Latin-1) com troca de símbolos sem glifo.
- **Após gerar, contamos as páginas com `pymupdf`.** Se passar do limite, primeiro retiramos as
  figuras opcionais; se ainda passar, `ReportTooLongError` — o limite é rígido no enunciado e
  preferimos falhar alto a entregar 4 páginas.

### Nomeação

`ficha.report.naming.delivery_basename(integrantes)` deriva `AtividadeI_<Sobrenome1>_<Sobrenome2>…`
(último sobrenome de cada integrante, sem acentos). Enquanto os integrantes forem placeholder,
o nome é `AtividadeI_SOBRENOMES` e o checklist final do notebook lembra de preencher.

### CLI

`ficha` (typer) expõe comandos finos que só compõem o pacote (`ingest`, `extract`, `audit`,
`cost`, `report`, `smoke`). O `smoke` roda o pipeline inteiro com FakeLLM em um diretório
temporário: é a demonstração de 30 segundos de que tudo está ligado.

## Alternativas consideradas

- *Notebook escrito à mão*: mais natural no Colab, mas não versionável de forma legível e sem
  teste de aceite automático.
- *Relatório via `nbconvert --to pdf`*: exige LaTeX, não controla o número de páginas e
  mistura código com texto.
- *Relatório em Markdown → PDF (pandoc)*: dependência externa pesada; mantivemos um template
  Markdown (`docs/relatorio/RELATORIO_TEMPLATE.md`) apenas como roteiro de revisão do texto.
- *Somente XLSX*: o CSV é o formato mais portável; entregamos os dois pelo mesmo custo.

## Consequências

- Mudou algo no notebook? Edite `scripts/build_notebook.py` e rode `make notebook`, nunca o
  `.ipynb` diretamente (as edições manuais seriam sobrescritas).
- Um relatório longo demais falha a geração; o grupo precisa encurtar o texto (o que é
  exatamente o que o enunciado pede).
- A figura só vai para o PDF se couber; no notebook ela sempre aparece.
