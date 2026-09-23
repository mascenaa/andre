# ADR 0002 — Seleção de contexto: três estratégias, chunk 1200/200, consultas em português

**Status:** aceito · **Data:** 2026-09-23 · **Seção do enunciado:** 4.1 (e insumo de 4.4c e 4.5)

## Contexto

Um artigo médio da bibliografia tem ~25 páginas (476 páginas / 19 artigos) e ~74 mil caracteres
(1,4 M / 19), algo como 18 mil tokens. Mandar o artigo inteiro é caro (Seção 4.5), não cabe
com folga no contexto de um Qwen2.5-3B na T4 e piora a extração: o modelo pequeno se perde em
texto longo. É preciso escolher **que pedaço** de cada artigo vai para o prompt, e a escolha
precisa ser justificada e, idealmente, medida.

Dois requisitos vêm de outras frentes:

- a auditoria de fidelidade (4.4a) compara `evidencia.trecho` com **o texto enviado**, então
  todo pedaço enviado precisa apontar para a página exata de onde saiu;
- a verificação 4.4c pede fichas extraídas a partir de **duas estratégias diferentes**, então
  precisamos de mais de uma, intercambiáveis.

Antes da seleção, o texto sai do PDF limpo (`ficha.ingest.clean`): hifenização de fim de linha
desfeita com heurística baseada no vocabulário do próprio documento, parágrafos reconstruídos,
cabeçalho/rodapé repetido e número de página removidos, NFKC e ligaturas. A limpeza é
conservadora e idempotente porque a fidelidade é medida contra ela.

## Decisão

Implementamos as três estratégias do enunciado atrás do mesmo contrato (`ContextSelector`),
selecionáveis por nome (`first_pages`, `keyword`, `semantic`) em `ficha.select.registry`:

1. **`first_pages`** — as 3 primeiras páginas inteiras. Linha de base barata e previsível.
2. **`keyword`** — janelas de até 1 500 caracteres depois de cabeçalhos de seção encontrados por
   regex tolerante a numeração (`3 Methods`, `3. Methodology`, `III. APPROACH`, `Materials and
   Methods`, `Threats to Validity`...), com orçamento total de 6 000 caracteres distribuído por
   prioridade: abstract → **limitações** → método → dados → resultados/avaliação → discussão →
   conclusão. Fallback declarado: sem cabeçalhos, vira `first_pages(2)`.
3. **`semantic`** — o artigo é segmentado em chunks de **1 200 caracteres com 200 de
   sobreposição**; para cada uma de 8 consultas em português (uma ou duas por campo da ficha,
   incluindo literalmente "como o desempenho foi medido" e "o que este trabalho não consegue
   fazer") pegamos os *k* chunks mais próximos por cosseno, unimos sem duplicatas e
   **reordenamos por (página, offset)**.

Invariantes comuns: nenhum chunk atravessa página; `page.text[start:end] == chunk.text`;
`Context.render()` prefixa cada pedaço com `[p. N]`; `Context.params` registra todos os
parâmetros (n páginas, janela, orçamento, k, tamanho, sobreposição, embedder).

### Por que 1 200 / 200 caracteres

- Em inglês acadêmico, ~4 caracteres por token: 1 200 caracteres ≈ **300 tokens**, 200 ≈ **50
  tokens**. É o tamanho de um parágrafo típico de artigo — a unidade em que autores declaram
  uma limitação, descrevem um conjunto de dados ou reportam uma métrica.
- A frase média em artigos científicos tem 25–30 palavras (~150–180 caracteres). Um chunk
  comporta ~7 frases; a sobreposição de 200 comporta **uma frase inteira**, de modo que uma
  afirmação cortada no fim de um chunk aparece completa no seguinte. O corte prefere fim de
  parágrafo, depois fim de sentença, e o início do chunk seguinte prefere começo de sentença.
- Menor (ex.: 400) fragmenta a evidência: o trecho que o modelo cita fica sem o contexto que o
  torna defensável. Maior (ex.: 3 000) dilui a similaridade — o vetor de um chunk longo é a
  média de vários assuntos, e a consulta de "limitação" deixa de achar o parágrafo certo.
- O modelo de embedding padrão (MiniLM) trunca em 128 *word pieces* por padrão da biblioteca
  (≈ 90 palavras); o chunk é mais longo que isso. Aceitamos: o início do chunk (que tende a ser
  começo de sentença/parágrafo) domina o vetor, e o custo de chunks menores (mais fragmentação
  para o modelo gerador) é maior que o ganho de recuperação. Fica registrado como limite.

### Por que consultas em português e embedder multilíngue

As consultas são escritas pelo grupo em português (é a língua em que pensamos os campos da
ficha), e os artigos estão em inglês. Um embedder só de inglês trataria a consulta como ruído.
`paraphrase-multilingual-MiniLM-L12-v2` projeta as duas línguas no mesmo espaço, roda em CPU
em minutos (FAQ do enunciado) e é o padrão de `Settings.embedding_model`. Nos testes e no modo
de ensaio usamos `HashingEmbedder` (bag-of-words + n-gramas de caracteres sem acento, sem
download): ele aproxima cognatos (`limitações`/`limitations`) mas não entende paráfrase — é um
substituto funcional, não semântico.

Achado do ensaio com PDFs sintéticos: a consulta literal "o que este trabalho não consegue
fazer" sozinha, com o MiniLM, trouxe a seção de limitações em só 1 de 3 artigos que a têm — a
formulação é indireta demais para um modelo pequeno. Por isso cada campo tem uma segunda
consulta explícita ("limitações declaradas pelos autores e ameaças à validade do estudo").

### Por que reordenar por página

Os chunks escolhidos saem do ranqueamento fora de ordem. Reordenados por (página, offset), o
modelo lê um texto que flui como o artigo, os marcadores `[p. N]` aparecem em ordem crescente
(o que ajuda o modelo a devolver `evidencia.pagina` correto) e a comparação entre execuções
não depende de empates no score. Chunks vizinhos escolhidos que se sobrepõem são unidos, para
não mandar os mesmos 200 caracteres duas vezes (custo da Seção 4.5).

## Alternativas consideradas

- **Artigo inteiro**: é a "alternativa ingênua" da Seção 4.5; entra só como linha de base de
  custo.
- **Uma só estratégia**: mais simples, mas inviabiliza a verificação 4.4c e esconde o
  trade-off. Manter as três custa pouco (mesmo contrato) e transforma a escolha em medição.
- **Chunk por seção detectada** (sem janela fixa): depende totalmente da detecção de
  cabeçalhos, que falha em PDFs com cabeçalho em fonte/linha estranha; o `keyword` já cobre
  esse caminho com fallback.
- **Chunk por tokens do tokenizador do Qwen**: mais preciso, mas acopla a seleção ao modelo
  gerador; caracteres são estáveis entre modelos e a conversão (~4 caracteres/token) é
  declarada.
- **Banco vetorial (FAISS/Chroma)**: permitido, mas desnecessário — são dezenas de chunks por
  artigo e a busca é dentro do próprio artigo; um produto de matrizes em NumPy basta.

## Consequências e limites de cada estratégia

- **`first_pages`**: barata e estável, mas cega para o fim do artigo. Tende a devolver
  `limitacao = null` mesmo quando os autores declaram limitações na seção 6 — é exatamente o
  contraste que a 4.4c deve mostrar.
- **`keyword`**: ótima quando o artigo tem seções convencionais; frágil em artigos com títulos
  criativos ("What we could not do"), limitações embutidas na Discussão sem cabeçalho próprio,
  ou cabeçalhos que a extração colou ao parágrafo. O orçamento de 6 000 caracteres pode cortar
  seções longas de método. O fallback é declarado em `params["fallback"]`.
- **`semantic`**: acha limitações sem cabeçalho e métricas espalhadas, mas depende da
  qualidade do embedder e das consultas; com `top_k` por consulta e 8 consultas, o contexto
  pode chegar a dezenas de chunks em artigos longos — `max_chars` limita o orçamento cortando
  os de menor score. Chunks de páginas diferentes perdem a continuidade da argumentação.
- Hifenização que atravessa páginas não é desfeita (juntar mudaria a página da evidência).
- A escolha final entre as três é decidida pelos números da 4.4c (fidelidade, taxa de `null`
  correto e divergência entre fichas), não por intuição.

## Revisão após execução real (19 artigos, Qwen2.5-3B)

### Problema observado

`limitacao` veio `null` em 18/19 fichas com `semantic` (orçamento 8 000 caracteres) e em 19/19
com `first_pages`. Antes de mexer no prompt, medimos se a **entrada** tinha a informação
(`ficha.select.diagnostics`, offline, sem modelo): o regex `LIMITATION_SENTENCE_RE` encontra
63 frases com marcas de limitação em 15 dos 19 artigos, e contamos quantas chegam ao contexto
(chave = primeiros 80 caracteres normalizados). O modelo não pode declarar o que não recebe.

### O que a medição mostrou

1. **As frases estão espalhadas, não concentradas no fim.** Só 11 das 63 estão dentro de
   seções de fechamento (Discussion/Conclusions/Limitations/Outlook) e 5 estão depois de
   `References` (agradecimentos). As outras 47 estão no corpo do artigo. Uma estratégia só por
   cabeçalhos de fechamento tem, portanto, teto de ~17%.
2. **O regex de medição é ruidoso.** Revisamos as 63 frases à mão: só ~18 são limitações que os
   autores admitem sobre o próprio trabalho. As demais são "we note that ...", "cannot be ..."
   descritivos, conselhos ("we caution the reader") e agradecimentos ("we acknowledge the use
   of the cluster"). Precisão estimada ≈ 30%. Por isso reportamos as duas medidas: o regex
   inteiro e o subconjunto revisado (18 frases, em 8 artigos).
3. **O recall cresce com o orçamento, quase linearmente.** Sem um sinal específico, cobrir
   limitações espalhadas é mandar mais texto.

Recall com o regex inteiro (63 frases em 15 artigos). Tokens ≈ caracteres / 4. Artigo médio:
75 787 caracteres (≈ 19 mil tokens).

| estratégia | artigos cobertos (de 15) | frases no contexto | recall | chars médios | tokens/artigo |
|---|---|---|---|---|---|
| first_pages (3 p.) | 6 | 8 | 12,7% | 10 411 | ≈ 2 600 |
| keyword (6k) | 4 | 4 | 6,3% | 4 910 | ≈ 1 230 |
| semantic (8k) | 6 | 9 | 14,3% | 7 418 | ≈ 1 850 |
| hybrid 12k, só seções | 7 | 13 | 20,6% | 11 413 | ≈ 2 850 |
| **hybrid 12k + 10 pistas (padrão)** | **9** | **19** | **30,2%** | **11 445** | **≈ 2 860** |
| hybrid 16k + 10 pistas | 11 | 21 | 33,3% | 14 895 | ≈ 3 720 |
| hybrid 20k + 10 pistas | 15 | 28 | 44,4% | 17 900 | ≈ 4 480 |
| hybrid 40k + 10 pistas (candidatos esgotados) | 15 | 32 | 50,8% | 21 925 | ≈ 5 480 |

Recall no subconjunto revisado à mão (18 limitações reais em 8 artigos):

| estratégia | artigos cobertos (de 8) | frases | recall |
|---|---|---|---|
| first_pages | 3 | 3 | 16,7% |
| keyword | 2 | 2 | 11,1% |
| semantic | 1 | 2 | 11,1% |
| hybrid 12k, só seções | 2 | 3 | 16,7% |
| **hybrid 12k + 10 pistas** | **5** | **10** | **55,6%** |
| hybrid 20k + 10 pistas | 7 | 12 | 66,7% |

`tail_pages` (as últimas páginas antes de `References`) não mudou nada com 12k (30,2%) e foi
desligado: como é a fonte de menor prioridade, o orçamento acaba antes de chegar nela.

### Decisão

Nova estratégia **`hybrid`** (`ficha.select.hybrid.HybridSelector`), registrada no
`build_selector`. Ela une as fontes com orçamento contado só em caracteres **novos**, nesta
prioridade:

(a) janelas após cabeçalhos de limitação (`Limitations`, `Caveats`, `Threats to Validity`,
`Current limitations`);
(b) até 10 janelas curtas (≤ 600 caracteres: a frase e as vizinhas) em torno de **pistas
lexicais** de limitação (`LIMITATION_CUE_RE`: limitation, drawback, caveat, shortcoming, beyond
the scope, future work, do/does not account/capture/generalize, may not generalize, small
sample, lack of...), só no corpo do artigo, pistas fortes primeiro;
(c) os chunks semânticos por score;
(d) janelas de Discussion/Conclusions/Summary/Outlook/Future Work;
(e) opcionalmente, as últimas páginas.

Os trechos são unidos quando se sobrepõem e ordenados por (página, início). `params` registra
`chars_por_fonte` e os candidatos que ficaram fora do orçamento. O padrão é **12 000 caracteres**
(≈ 2 900 tokens, 6,6× mais barato que o artigo inteiro e +54% sobre o `semantic`). O
`KeywordSectionSelector` ganhou os cabeçalhos de fechamento que faltavam (`Summary and
Discussion`, `Comments and Conclusions`, `Discussion and Conclusions` sem número, `Outlook`,
`Future Work`) e o parâmetro opcional `sections` (retrocompatível).

**A meta de ≥ 50% das frases e ≥ 12/15 artigos não é atingida com 12k.** Com o regex inteiro,
ela só é atingida quando todos os candidatos entram (≈ 22k caracteres, ≈ 5 500 tokens por
artigo, 29% do texto). A cobertura de artigos (≥ 12/15) é atingida com **20k** (15/15, recall de
44%, ≈ 4 500 tokens). O trade-off, ligável por `FICHA_HYBRID_MAX_CHARS`:

- 12k: 2 860 tokens/artigo, 9/15 artigos, 5/8 artigos com limitação real coberta;
- 20k: 4 480 tokens/artigo (+57%), 15/15 artigos, 7/8 com limitação real coberta.

### Cuidado com circularidade

As pistas de **seleção** e o regex de **medição** compartilham cinco termos (limitation,
caveat, drawback, shortcoming, beyond the scope). Parte do ganho das pistas é, portanto,
medição que se autoconfirma. Controle: medimos o recall só nas 54 frases do regex de medição
que **não** têm nenhuma pista de seleção. Nelas, as pistas não ajudam com 12k (20,4% com ou
sem pistas), e o ganho vem do orçamento (37,0% com 20k). O ganho real das pistas está nas
limitações explícitas — justamente as que a ficha precisa —, e isso aparece no subconjunto
revisado à mão (16,7% → 55,6%). A validação final é a jusante: taxa de `limitacao` não nula
**com trecho fiel** na 4.4a, comparada entre `semantic` e `hybrid` na 4.4c.

### Limites que continuam

- `References` no meio do PDF (formato Nature, com métodos depois das referências, como em
  `11_Jarolim`) corta a busca de cabeçalhos e pistas nesse ponto.
- Pistas lexicais pegam limitações escritas com esse vocabulário; admissões implícitas
  ("results should be interpreted with care") escapam.
- Artigos de revisão (`09_AsensioRamos`, `18_Karniadakis`) discutem limitações **de outros
  trabalhos**; a janela entra no contexto, e cabe ao prompt (instrução de abstenção) não
  atribuí-las aos autores.
