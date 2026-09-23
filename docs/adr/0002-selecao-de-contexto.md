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
