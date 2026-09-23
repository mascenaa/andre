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

## Calibração após execução real (2026-09-23)

**Dados.** Pipeline real com Qwen2.5-3B-Instruct, 19 artigos, 5 execuções (`data/runs/`).
Comparando as estratégias `semantic` × `first_pages` (19 pares), a similaridade média por
campo foi: problema 0.49, dados 0.46, metodo 0.49, metrica 0.49, evidencia.trecho 0.48,
evidencia.pagina 0.16, limitacao 0.95. Entre repetições (mesma entrada, decodificação gulosa):
19/19 fichas idênticas (similaridade 1.00 em todos os campos).

**Problema.** A regra v1 deu alta 0 / media 2 / baixa 17, quase sempre pelo motivo "6 campos
mudaram entre estratégias". Duas entradas diferentes produzem, por construção, paráfrases
diferentes e evidências de páginas diferentes — o critério media **redação**, não
confiabilidade. Já a instabilidade com a *mesma* entrada é informativa, e ali a comparação
exata continua valendo.

**Decisão (regra v2, padrão).** `ConfidenceRule.input_effect_mode`:

- `"strict"` (v1, preservada e reproduzível com `RULE_V1`): toda diferença conta, também
  entre estratégias;
- `"categorical"` (v2, `RULE_V2`, padrão): entre **estratégias** só conta a discordância
  categórica em `limitacao` — `null` de um lado, texto do outro (`DiffReport.n_categorical_changes`).
  Essa discordância é de conteúdo ("os autores declaram limitação?"), não de redação, e
  rebaixa para MEDIA. Entre **repetições** a comparação continua exata.

Os outros critérios não mudam: trecho não encontrado → BAIXA (pegou 08_Leka: o trecho era a
frase do exemplo few-shot, score 0.47 — invenção detectada); página errada → BAIXA (11_Jarolim e
D04_Jiao: trecho existe, página declarada errada).

**Heurística de limitação.** O veredito `null_suspeito` (limitação null, mas a palavra
"limitation" aparece no contexto) **não rebaixa** — e na implementação nunca rebaixou; agora
isso está declarado. A leitura confirmou falsos positivos: um título "Prospects, Scope, and
Limitations" e um parágrafo sobre "limitations of generative AI in academic writing" — a
palavra aparece sem ser uma limitação declarada do próprio trabalho. `preenchida_sem_suporte`
(limitação preenchida sem nenhum termo de limitação no contexto) continua limitando a MEDIA: é
o erro grave do enunciado e não sofre desse falso positivo (a ausência total do vocabulário é
um sinal forte).

**Alternativa rejeitada.** Manter a comparação de texto livre entre estratégias, com um limiar
de similaridade (ex.: "mudou" se `ratio < 0.4`). Qualquer valor seria arbitrário: paráfrases
corretas em pt-BR do mesmo conteúdo ficam em ≈0.4–0.6 de `ratio`, a mesma faixa de conteúdos
de fato diferentes, então o limiar não separa "reescreveu" de "discorda".

**Comparabilidade.** `confidence_by_rule(primary, rep, alt, {"v1": RULE_V1, "v2": RULE_V2})`
produz a confiança e os motivos por artigo sob as duas regras lado a lado;
`confidence_distribution(fichas)` dá a distribuição de cada uma. O texto de cada regra
(`describe()`) inclui o modo ativo e esta justificativa numérica.

## Verificação e) Vazamento de exemplos few-shot (2026-09-23)

**Contexto.** Na execução híbrida, 06_Barnes_2016 (previsão de explosões solares) saiu com
`limitacao` = "Os rótulos vêm de chargebacks, então fraudes nunca contestadas pelos clientes
ficam fora do ground truth": a limitação do **exemplo few-shot** de fraude em cartão. A
fidelidade passa (o trecho é do artigo, na página certa) e a heurística de limitação também
(o contexto tem vocabulário de limitação). É a "invenção convincente" que a rubrica premia
detectar — e nenhuma das verificações a) a d) a pegava. O mesmo vazamento já aparecia em
`evidencia.trecho` (08_Leka, execução semantic).

**Decisão.** `ficha.audit.leakage.check_fewshot_leakage(record)` compara cada campo textual da
ficha (`problema, dados, metodo, metrica, limitacao, evidencia.trecho`) com os exemplos de
`ficha.prompts.FEW_SHOT_EXAMPLES`, por dois métodos:

1. **Similaridade com o campo homólogo** do exemplo, após `normalize_for_match`: o máximo entre
   `fuzz.ratio` e `fuzz.partial_ratio` (este só quando a string mais curta tem ≥ 40
   caracteres — com "AUPRC" contra a métrica do exemplo ele daria 1.0 por coincidência de
   sigla). O `evidencia.trecho` também é comparado com o **texto de entrada** do exemplo
   (`partial_ratio`), porque copiar qualquer frase do exemplo é vazamento. Vazou se ≥ **0.80**.
   Calibração nas execuções reais (450 comparações campo × melhor exemplo, fichas sem
   vazamento): média 0.45, mediana 0.46, máximo 0.77. As cópias ficam em 0.93–1.00
   (Barnes: 1.00 em `limitacao`; Leka: 0.97–1.00 em dados, metodo, metrica e trecho). O
   limiar fica no vão entre as duas populações.
2. **Termos de domínio exclusivos dos exemplos** (`EXAMPLE_MARKER_TERMS`: fraude, chargeback,
   cartão de crédito, emissor, transações; supermercados, varejo, vendas, estoque, promoções,
   feriados, WMAPE, suavização exponencial) presentes na ficha **sem** o equivalente em inglês
   no texto enviado. Esse método pega a **contaminação parcial**, que a similaridade do campo
   inteiro não pega. Os exemplos foram escritos de propósito em domínios alheios aos artigos
   (fraude e varejo contra física solar), então esses termos funcionam como marcadores. Se o
   artigo falar de fato de vendas ("sales"), o termo é legítimo e não conta.

`None` nunca vaza (abster-se não é copiar). Na regra de confiança, v1 e v2, **qualquer**
vazamento limita a **BAIXA**, com um motivo por campo: "campo limitacao copiado do exemplo
few-shot (similaridade 1.00)" e "campo dados contém 'supermercados', termo do exemplo few-shot
ausente do texto enviado". Os dois métodos podem ser desligados (`check_leakage`,
`check_leakage_terms`) e o limiar é `leakage_threshold`.

**Resultado nas execuções reais** (medido; leitura de `data/runs/`, sem escrita):

| execução | fichas com vazamento | casos |
|---|---|---|
| semantic v_full t0 (rep1 e rep2) | 2/18 | 08_Leka (exemplo 2 inteiro: dados, metodo, metrica, trecho); 09_AsensioRamos ("magnetogramas ... de **supermercados brasileiros**": similaridade 0.77, só o termo pega) |
| semantic v_sem_fewshot t0 | 0/18 | controle: sem exemplos no prompt, nenhum falso positivo |
| first_pages v_full t0 | 3/18 | 08_Leka e D03_Sun (métrica "WMAPE ... 13 semanas"), 16_Nishizuka ("comparado com a suavização exponencial") |
| semantic v_full t0.7 (subconjunto) | 0/6 | — |
| hybrid v_full t0 | 1/16 (execução em andamento) | 06_Barnes (`limitacao` do exemplo 1) |

**Na comparação de prompts**, `RunStats.rate_fewshot_leak` aparece lado a lado. Few-shot
melhora formato e fidelidade, mas abre a porta para copiar o exemplo. A variante sem few-shot
tem 0 por construção e serve de controle de falso positivo. A taxa **não** entra no critério
de vitória, que foi declarado antes de ver os dados; mudá-lo agora seria escolher por
intuição. Ela entra na confiança de cada ficha e na discussão do relatório.

**Alternativas rejeitadas.**
- *Só o trecho* (o check ad hoc anterior): não pega Barnes (`limitacao`) nem a contaminação
  parcial.
- *Só similaridade*: não pega 09_AsensioRamos (0.77) nem 16_Nishizuka (0.63). Baixar o limiar
  para alcançá-los cruzaria a população legítima (máximo 0.77).
- *Embeddings*: aproximariam paráfrases legítimas do mesmo tipo de campo ("dados de X de 2010
  a 2020; volume não informado") e dariam falsos positivos. O que caracteriza a cópia é a
  forma literal ou o vocabulário de outro domínio, não o sentido.

**Consequência.** A lista de marcadores está acoplada aos exemplos. Um teste confere que cada
termo existe no seu exemplo, então mudar os exemplos obriga a revisar a lista.
