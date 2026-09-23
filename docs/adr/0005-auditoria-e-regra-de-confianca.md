# ADR 0005 — Auditoria quantificada e regra declarada de confiança

**Status:** aceito · **Data:** 2026-09-23 · **Código:** `src/ficha/audit/**`, `src/ficha/cost/**`

## Contexto

A auditoria (Seção 4.4) vale 30% da nota e o julgamento (quais fichas não defenderíamos) mais
15%. O enunciado exige quatro verificações **quantificadas** (fidelidade, estabilidade, efeito
da entrada, efeito da temperatura), uma medição obrigatória entre duas versões do prompt (4.2),
um campo `confianca` atribuído por **regra declarada**, e custo com premissa visível (4.5).
Tudo o que derruba a nota é o que não se mede: "o modelo acertou", confiança "no olho",
prompt escolhido por intuição.

## Decisão

### 1. Fidelidade contra o texto enviado, não contra o PDF

`check_fidelity` procura `evidencia.trecho` em `ExtractionRecord.context_text` — o texto
**exato** que o modelo recebeu (`Context.render()`). O modelo não viu o resto do PDF; um
trecho que está no artigo mas fora do contexto enviado foi "lembrado" ou reconstruído, e não
pode sustentar a ficha. Comparar contra o PDF inteiro aprovaria justamente esse caso.

### 2. Normalização tipográfica declarada

Antes de qualquer comparação, os dois lados passam por `normalize_for_match`: Unicode NFKC
(desfaz ligaduras `ﬁ`, reticências, NBSP), remoção de caracteres invisíveis (hífen suave,
largura zero), aspas tipográficas → retas, hífens/travessões/menos → `-` sem espaços ao redor,
`casefold`, espaços colapsados, sem espaço antes de pontuação de fechamento nem depois de
abertura. Nada disso muda palavras, números ou ordem: paráfrase continua diferente.

### 3. Exato primeiro; depois `partial_ratio` com limiar 0.90

1. Substring exata após normalização → `method="exact"`, `score=1.0`.
2. Senão, `rapidfuzz.fuzz.partial_ratio(trecho, contexto)/100` → `method="fuzzy"`.
   `partial_ratio` é a melhor razão de Levenshtein entre o trecho e **qualquer janela do
   contexto do mesmo tamanho**, que é exatamente a pergunta "este trecho é um pedaço contíguo
   do texto?". A razão simples (`ratio`) penalizaria a diferença de tamanho entre trecho e
   contexto; `token_set_ratio` ignoraria a ordem das palavras e aprovaria colagens.
3. `found = score ≥ 0.90`. Em 200 caracteres, 0.90 admite ~20 edições — uma palavra
   hifenizada na extração, uma vírgula, um número de citação omitido — mas não uma frase
   reescrita. Nos testes: trecho com erro de limpeza → ~0.97 (passa); paráfrase → ~0.51;
   trecho inventado do `FakeLLM` → ~0.46. A folga entre 0.51 e 0.90 é grande, então o
   resultado não é sensível ao valor exato do limiar. O limiar é parâmetro
   (`ConfidenceRule.fidelity_threshold`) e aparece no texto da regra.
4. Página: o contexto é dividido pelos marcadores `[p. N]`. Se o trecho está (≥ limiar) em
   algum bloco da página declarada, `page_found` é ela (trecho repetido em duas páginas não é
   penalizado); senão, é a página do bloco onde caiu a melhor ocorrência. `page_ok` exige
   `page_found == evidencia.pagina`.

Heurística complementar para o erro mais grave do enunciado (limitação inventada):
`check_limitacao_support` marca `preenchida_sem_suporte` quando `limitacao` veio preenchida e o
texto enviado não contém **nenhum** termo de limitação (`limitation(s)`, `limited`,
`shortcoming`, `drawback`, `caveat`, `threats to validity`, `future work/research`,
`we acknowledge`, `weakness`). É indício, não prova — por isso só rebaixa, nunca promove.

### 4. Estabilidade, entrada e temperatura: um motor de diferenças

`diff_runs(A, B)` pareia por `arquivo` e compara os campos `problema, dados, metodo, metrica,
limitacao, evidencia.trecho, evidencia.pagina` (`confianca` não vem do modelo):

- **igual** = strings iguais após normalização; `pagina` igual por número;
  ambos `null` → iguais; só um `null` → diferentes (similaridade 0: "declara limitação" versus
  "não declara" é divergência de conteúdo, não de redação);
- **similaridade** = `fuzz.ratio/100` nas formas normalizadas (1/0 para página);
- artigos sem ficha válida num dos lados contam em `n_unpaired` — nem iguais, nem diferentes.

Três números quantificam a divergência: **taxa de fichas idênticas** (todos os campos iguais),
**taxa de mudança por campo** (responde "quais campos mudam mais", ordenado por
`most_unstable_fields`) e **similaridade média por campo** (quanto mudam). Por artigo, o nº de
campos que mudaram alimenta a regra de confiança.

- 4.4b `stability_report(rep1, rep2)`: mesma configuração; avisa se strategy/prompt/modelo/
  temperatura/seed diferirem.
- 4.4c `input_effect_report(A, B)`: além do diff, fidelidade de cada estratégia e recomendação
  por critério declarado: maior fidelidade → maior taxa de página correta → menos falhas de
  parse → menos tokens de entrada.
- 4.4d `temperature_report(t0, t_alt)`: restringe `t0` ao subconjunto repetido; a escolha de
  `t0` **se sustenta** se, nesse subconjunto, sua taxa de fidelidade e sua taxa de JSON válido de
  primeira forem ≥ às da temperatura alternativa.

### 5. Critério de vitória entre prompts (4.2)

Para cada variante: JSON válido de primeira, parse ok/reparado/falhou (denominador = todas as
chamadas), `limitacao` null (denominador = fichas válidas), **null quando devido** (fichas de
artigos sem limitação declarada — pelo gabarito conferido à mão, ou pela heurística de
vocabulário se não houver gabarito, o que fica registrado) e fidelidade. Vencedora por ordem
lexicográfica: (1) mais JSON válido de primeira; (2) empate → maior fidelidade; (3) empate →
mais null quando devido. Critério sem dados (sem nenhum artigo em que null é devido) é pulado.
Empate total → fica a variante de referência, com a conclusão explícita "a técnica não mudou
nada nestes números".

### 6. Regra de confiança

O nível é o **menor** entre os limites abaixo (`ConfidenceRule.describe()` gera o texto):

| Condição | Limite |
|---|---|
| extração falhou (nenhuma ficha válida pelo esquema) | BAIXA |
| trecho não encontrado no texto enviado (`score < 0.90`) | BAIXA |
| trecho encontrado em página diferente da declarada | BAIXA |
| mais de 2 campos mudaram entre repetições **ou** entre estratégias | BAIXA |
| 1–2 campos mudaram em alguma das comparações | MEDIA |
| alguma comparação não pôde ser feita (sem repetição / sem ficha do outro lado) | MEDIA |
| `limitacao` preenchida sem vocabulário de limitação no contexto | MEDIA |
| nada acima: evidência na página declarada e 0 campos mudaram nas duas comparações | ALTA |

"Não verificado" não recebe ALTA: a ausência de evidência de instabilidade não é evidência de
estabilidade. `ParseStatus.REPAIRED` não rebaixa por si só (a ficha passou pelo esquema), mas é
reportado nas taxas da 4.2. Cada ficha carrega a lista de **motivos** que a limitaram
(ex.: `trecho não encontrado no texto enviado (score 0.46 < 0.90): possível trecho inventado`).
Extração falha vira linha `EXTRAÇÃO FALHOU` com BAIXA — a tabela sempre tem as 19 linhas.
As fichas BAIXA, com seus motivos, são a lista "fichas que não defenderíamos".

### 7. Custo (4.5)

`usd = tokens / 10⁶ × preço_por_Mtok`, separado em entrada e saída, com a premissa
(`PricePremise`, lida de `Settings`: US$ 3/Mtok de entrada, US$ 15/Mtok de saída, fonte
declarada) impressa em toda tabela e dicionário. O custo **real** soma o `Usage` de **todas**
as chamadas feitas (repetição de estabilidade, outra estratégia, outra temperatura, outra
variante de prompt). A alternativa **ingênua** manda o artigo inteiro em cada chamada, com o
mesmo nº de chamadas por artigo e uma estimativa declarada de 300 tokens de saída por ficha,
ignorando o limite de contexto (é o custo *se* coubesse). Reportamos a razão em dólares e a
razão de tokens de entrada (esta independe da premissa de preço).

## Alternativas consideradas

- *Fidelidade contra o PDF inteiro*: aprovaria trechos que o modelo não viu.
- *Fidelidade só por substring exata*: reprovaria trechos corretos por um hífen de fim de linha.
- *Similaridade semântica (embeddings) para fidelidade*: aprovaria paráfrases — exatamente o
  que queremos pegar. Evidência precisa ser literal.
- *Confiança como média ponderada de escores*: pesos arbitrários e difíceis de defender; a regra
  por limites é auditável linha a linha.
- *Pedir ao modelo a confiança*: proibido pelo enunciado e sem relação com as verificações.

## Consequências

- Cada número do relatório vem de uma função testada (`tests/test_audit_*.py`,
  `tests/test_cost_*.py`) e é reproduzível a partir de `records.jsonl`.
- A heurística de limitação pode dar falso positivo (autor declara limitação sem essas
  palavras) — o efeito máximo é rebaixar para MEDIA, e o motivo fica explícito.
- O limiar 0.90 e os limites de campos são parâmetros; mudá-los muda o texto da regra, que é
  gerado a partir deles.
