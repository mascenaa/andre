# ADR 0003 — Engenharia de prompt: técnicas identificáveis, variantes e temperatura

**Status:** aceito · **Data:** 2026-09-23

## Contexto

A Seção 4.2 exige que o prompt de extração use, **de forma identificável**, instrução explícita
com formato exato de saída, papel, delimitadores, few-shot (com um caso `null`) e instrução de
abstenção; permite cadeia de raciocínio desde que o conflito com JSON puro seja resolvido; e
exige **duas versões que difiram em uma técnica**, comparadas com números. A rubrica derruba
"escolher o prompt final por intuição". Também pede temperatura declarada e justificada.

## Decisão

### 1. Uma técnica = um bloco com cabeçalho fixo (`src/ficha/prompts/builder.py`)

| Técnica (`PromptFeatures`) | Bloco no prompt | Onde |
|---|---|---|
| papel (`role`) | `### PAPEL` | mensagem `system` (desligado → `system=None`) |
| instrução explícita (sempre) | `### INSTRUÇÃO` | `user` — o que extrair, campo a campo |
| formato de saída (sempre) | `### FORMATO DE SAÍDA` | `user` — schema JSON embutido + "SOMENTE o JSON" |
| abstenção (`abstention`) | `### REGRA DE ABSTENÇÃO` | `user` |
| cadeia de raciocínio (`chain_of_thought`) | `### RACIOCÍNIO` | `user` (e muda o formato) |
| few-shot (`few_shot`) | `### EXEMPLOS` | `user` |
| delimitadores (`delimiters`) | `### ARTIGO` + `<<<ARTIGO>>>`…`<<<FIM_ARTIGO>>>` | `user`, sempre por último |

Por que cabeçalhos: a presença de cada técnica vira **verificável por teste**
(`tests/test_prompts_builder.py` liga/desliga cada flag e checa o cabeçalho) e **visível no
relatório** (`PromptBuilder.markers_present()` vai para o `manifest.json` de cada execução,
junto com `template_sha`, a impressão digital do molde).

Detalhes que importam:

- **Instrução explícita.** Cada campo é descrito; `evidencia.trecho` deve ser **cópia literal**
  do texto enviado e `evidencia.pagina` o `N` do marcador `[p. N]` que o precede. O schema de
  `FichaExtraida` vai embutido (sem os `title` do pydantic e com `limitacao` obrigatória na cópia
  do prompt, para que o modelo sempre escreva a chave — com `null` explícito).
- **Delimitadores.** O artigo entra exatamente como `Context.render()` o devolve, entre
  `<<<ARTIGO>>>` e `<<<FIM_ARTIGO>>>`, com o aviso de que nada ali dentro é ordem (defesa contra
  frases do artigo lidas como instrução). Os marcadores literais aparecem **uma vez só** no
  prompt — a prosa os descreve sem repeti-los —, e os exemplos usam marcadores próprios
  (`<<<EXEMPLO_ARTIGO>>>`), para não haver ambiguidade sobre qual é o artigo. Sem delimitadores,
  o artigo vem depois de uma linha `Artigo:` — é isso que a variante sem delimitadores mede.
- **Few-shot** (`src/ficha/prompts/examples.py`). Dois mini-artigos em inglês, em domínios
  diferentes dos artigos reais (fraude em cartão; previsão de demanda), no formato exato de
  `Context.render()`. O exemplo 1 declara uma limitação; o exemplo 2 **não declara** (tem só
  "future work", que não é limitação) e sua saída tem `"limitacao": null` e
  `"período e volume não informados"`. Os testes garantem que cada `trecho` é substring literal
  da entrada e que `pagina` bate com o marcador.
- **Abstenção.** "Se os autores não declaram limitação, `null`; nunca preencha com algo plausível;
  melhor `null` do que inventado." Para `dados`/`metodo`/`metrica` (que o schema exige como
  string), a instrução é escrever "não informado" — assim a ausência fica explícita e contável.

### 2. Cadeia de raciocínio × JSON puro

O conflito foi resolvido **pondo o raciocínio dentro do JSON**: com `chain_of_thought=True` o
formato pedido passa a ser `{"raciocinio": "...", "ficha": {...}}` (máx. 5 frases, dizendo em
que página está cada informação e se há limitação declarada). A saída continua sendo um único
objeto JSON validável; o parser desembrulha o envelope (`parse_completion(...,
chain_of_thought=True)`), guarda o raciocínio em `ParseResult.raciocinio` e **não** conta o
envelope como reparo. Os exemplos few-shot também passam a mostrar o envelope.

Trade-off: raciocinar antes pode ajudar a decidir a limitação (ler a página certa antes de
responder), mas custa tokens de saída (mais caro e mais lento na T4), aumenta o risco de a saída
ser truncada em `max_new_tokens` e, em modelo pequeno, aumenta a chance de JSON quebrado
(aspas não escapadas no texto livre). Por isso fica **desligada por padrão** (`v_full`) e é
medida como variante própria (`v_cot`), em vez de assumida como melhoria.

### 3. Variantes (`src/ficha/prompts/variants.py`)

Cada variante difere de `v_full` em **exatamente uma** técnica (garantido por teste):

- `v_full` — todas as técnicas exigidas, sem CoT;
- `v_sem_fewshot` — **par obrigatório** (`MANDATORY_PAIR = ("v_full", "v_sem_fewshot")`);
- `v_sem_delimitadores` — mede o efeito dos delimitadores;
- `v_cot` — `v_full` + raciocínio no envelope.

### 4. Como as versões são comparadas

Mesmo conjunto de artigos, mesma estratégia de seleção, mesmo modelo, mesma temperatura e
semente — só muda a variante. Para cada variante, a partir dos `ExtractionRecord`:

1. **Taxa de JSON válido de primeira** = `json_valid_first_try` / nº de registros. Definição
   (em `extract/parser.py`): o texto bruto passa em `json.loads` estrito **e** no schema, sem
   nenhum reparo textual (cercas, texto em volta, vírgula final, aspas tipográficas) nem
   estrutural (envelope inesperado, `pagina` em texto). Também reportamos a distribuição
   `ok/repaired/failed` e quais reparos foram necessários (`reparse(record).repairs`).
2. **Taxa de `null` correto** = entre os artigos que de fato não declaram limitação (gabarito
   anotado pelo grupo), a fração com `limitacao is None`; e, no sentido oposto, quantos `null`
   foram dados a artigos que declaram (abstenção excessiva).
3. **Discordância** = por artigo e por campo, se as fichas das duas variantes diferem (e em
   quê), usando a mesma comparação da verificação de estabilidade (4.4b).

A versão final é a que ganha nesses números; se não houver diferença, o relatório diz isso
("a técnica X não mudou nada neste caso"), com os números.

### 5. Temperatura 0.0 para a extração

Extração é uma tarefa de **fidelidade**, não de criatividade: queremos, para cada campo, a
continuação mais provável dado o texto — que é também a mais "colada" ao texto. Com
`temperature=0` o Qwen local roda em decodificação gulosa (`do_sample=False`), o que torna a
saída **determinística** para o mesmo prompt: isso dá estabilidade entre execuções (4.4b) e
reprodutibilidade (Seção 7), e faz com que diferenças entre variantes sejam atribuíveis ao
prompt, não ao sorteio. Amostrar (temperatura > 0) só acrescenta variância — e variância em
`limitacao` é exatamente o que produz limitações inventadas. A escolha não é assumida: a
verificação 4.4d repete um subconjunto com `temperature_alt = 0.7` e mede o que muda.

Detalhe do cliente local: forçamos `repetition_penalty=1.0` (o `generation_config` do Qwen usa
1.05), porque essa penalidade age sobre tokens que **já estão no prompt** — isto é, desestimula
copiar o trecho literal que `evidencia.trecho` exige.

## Alternativas consideradas

- *Prompt único, técnicas misturadas em prosa*: mais curto, mas impossível provar qual técnica
  está presente ou medir o efeito de retirar uma.
- *CoT livre antes do JSON* (texto + JSON no fim): quebra "somente JSON" e obriga o parser a
  extrair o JSON de texto livre — exatamente o reparo que queremos medir como falha.
- *Biblioteca de saída estruturada* (outlines, instructor, json_repair): proibida pelo
  enunciado; e esconderia a taxa de JSON malformado, que é resultado a relatar.
- *Temperatura baixa mas não nula (0.2–0.3)*: sem ganho claro para extração e perde o
  determinismo que sustenta a comparação de variantes.

## Consequências

- Cada execução registra variante, `features`, blocos presentes e `template_sha` no manifest,
  então qualquer número do relatório é rastreável até o prompt exato.
- O prompt com few-shot é ~60% mais longo (mais tokens de entrada, entra na conta de 4.5).
- A abstenção cobre só `limitacao` com `null`; os demais campos usam "não informado" porque o
  schema exige string — o relatório conta as duas formas de ausência separadamente.
- Com CoT desligado por padrão, a variante `v_cot` custa uma execução extra se o grupo quiser
  comentar "se valeu a pena", como o enunciado sugere.
