# ADR 0004 — Escolha do modelo: Qwen2.5-3B-Instruct em fp16 na T4

**Status:** aceito · **Data:** 2026-09-23

## Contexto

A Seção 4.3 aceita modelo local (família Qwen2.5-Instruct na GPU T4 gratuita do Colab, ~16 GB)
ou modelo por API, desde que declarado. A tabela do enunciado:

| Modelo | Cabe na T4? | Observação |
|---|---|---|
| Qwen2.5-1.5B-Instruct | com folga | rápido, mas erra bastante nesta tarefa |
| Qwen2.5-3B-Instruct | sim, em meia precisão | o melhor equilíbrio para esta atividade |
| Qwen2.5-7B-Instruct | só quantizado em 4 bits | mais capaz, mais lento |

Regra de bolso: 2 GB por bilhão de parâmetros em meia precisão, mais o espaço do contexto.
O enunciado também diz que erro de modelo pequeno é **matéria-prima** da auditoria (4.4), e
que, se JSON quebrado inviabilizar a tarefa, deve-se subir de tamanho e relatar a diferença.

## Decisão

**Padrão: `Qwen/Qwen2.5-3B-Instruct` em fp16**, `device_map="auto"`, temperatura 0.0
(gulosa — ver ADR 0003), via `ficha.llm.qwen_local.QwenLocalClient`
(`FICHA_MODEL_BACKEND=qwen_local`).

- Memória: 3,09 bi × 2 GB ≈ 6,2 GB de pesos + ~2,5 GB de folga para contexto/cache KV ≈ 8,7 GB
  (`estimate_memory_gb`), confortável nos ~15 GB livres da T4. O 1.5B caberia com mais folga,
  mas o próprio enunciado avisa que ele erra bastante; o 3B é o ponto de equilíbrio declarado.
- `fits_in_memory(params_bi, quantize_4bit, free_gb)` checa isso **antes** de carregar: o
  cliente falha cedo com mensagem acionável ("use `quantize_4bit=True` ou um modelo menor") em
  vez de estourar memória no meio da execução. `describe_gpu()` imprime nome e memória da GPU
  no início do notebook, para o relatório declarar o hardware.
- `Usage` usa a contagem **real** do tokenizador do Qwen (entrada = prompt já com o chat
  template; saída = tokens gerados) — é a base do custo (4.5).
- Sem papel (`system=None`), enviamos mensagem de sistema vazia: o chat template do Qwen2.5
  inseriria o papel padrão "You are Qwen, created by Alibaba Cloud...", contaminando a
  comparação "com papel × sem papel".

**Escalada: `Qwen/Qwen2.5-7B-Instruct` em 4 bits** (`FICHA_QUANTIZE_4BIT=true`,
`BitsAndBytesConfig(load_in_4bit=True, nf4, compute fp16)`), se a taxa de JSON válido do 3B —
mesmo depois de trabalhar o prompt — inviabilizar a tabela (FAQ do enunciado). Em fp16 o 7B
não cabe: 7,62 bi × 2 GB ≈ 15,2 GB só de pesos. Em 4 bits, ≈ 5,7 GB + folga. Se usado, o
relatório compara 3B × 7B nas mesmas fichas.

**Alternativa declarada: modelo por API** (`anthropic` ou `openai_compat` — Groq, Together,
Ollama local). Mesma interface `LLMClient`, chave só por variável de ambiente/Secrets do Colab
(`SecretStr`, nunca logada). Útil para o acréscimo "modelo local × API nas mesmas fichas", que
conta no critério de julgamento. Limitação a declarar: a API da Anthropic não aceita semente,
então `temperature=0` é quase — mas não garantidamente — determinística.

## Alternativas consideradas

- *1.5B*: mais rápido, mas a taxa de erro esperada tornaria a tabela pouco defensável; fica como
  ponto de comparação opcional.
- *7B em 4 bits como padrão*: mais capaz, porém mais lento na T4, com risco de estourar o tempo
  de sessão do Colab gratuito nas várias execuções que a auditoria exige (2 variantes × 2
  estratégias × repetição × temperatura alternativa).
- *Só API*: dispensaria GPU, mas perderia a medição honesta de modelo pequeno (e custaria
  dinheiro por execução repetida).

## Consequências

- Os erros do 3B (JSON malformado, trecho não literal, limitação inventada) são medidos, não
  escondidos: o parser registra cada reparo, a saída bruta é guardada e a auditoria (4.4)
  explica o padrão. Isso é o que o enunciado valoriza.
- Os testes nunca carregam o modelo: a lógica pura (mensagens, parâmetros de geração, memória)
  é testada sem `torch`, e a geração é testada com tokenizador/modelo falsos.
- O modo de ensaio (`FICHA_MODEL_BACKEND=fake`) roda o pipeline inteiro sem GPU antes de
  gastar a cota do Colab.
