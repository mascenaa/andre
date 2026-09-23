# Relatório — Atividade de Construção I (roteiro de revisão, ≤ 3 páginas)

> Este é o **esqueleto** do PDF gerado por `ficha.report.build_report_pdf` a partir de
> `ReportContent` (célula 12 do notebook). Os marcadores `{{...}}` são números que vêm da
> auditoria — **nenhum deve ser digitado à mão**. Use este roteiro para revisar o texto gerado
> antes da entrega: cada frase precisa ser defensável com um número do notebook.
> Limite: 3 páginas A4 (a geração falha com `ReportTooLongError` se passar).

**Título:** A ficha que você teria de defender
**Integrantes (primeira página):** {{integrantes}}
**Arquivos:** `AtividadeI_{{sobrenomes}}.ipynb`, `.csv`/`.xlsx`, `.pdf`

---

## 1. Estratégia de seleção do contexto (4.1)

- **Escolhida:** {{estrategia}} (alternativa: {{estrategia_alternativa}}).
- **Por quê:** {{justificativa — ver ADR 0002}}; segmentação de {{chunk_size}} caracteres com
  sobreposição de {{chunk_overlap}}, top-{{k}} trechos por consulta.
- **Tratamento do texto:** hifenização, quebras de parágrafo, cabeçalhos/rodapés repetidos.
- Tabela: estratégia × páginas médias × caracteres médios × tokens médios
  ({{first_pages_tokens}} / {{keyword_tokens}} / {{semantic_tokens}}).

## 2. Prompt: duas versões e a comparação (4.2)

| versão | técnicas |
|---|---|
| {{v1_nome}} | {{v1_features}} |
| {{v2_nome}} | {{v2_features}} |

- **O que muda:** {{describe_diff}}.
- **Números (mesmos {{n_artigos}} artigos):**
  JSON válido de primeira {{v1_json_ok}} vs {{v2_json_ok}};
  `null` correto quando não há limitação {{v1_null_ok}} vs {{v2_null_ok}};
  fichas que discordam {{n_discordam}} (campos: {{campos_discordantes}}).
- **Conclusão:** {{vencedor}} — decidido pelos números acima, não por intuição.

## 3. Modelo e temperatura (4.2–4.3)

- **Modelo:** {{modelo}} ({{precisao}}, {{gpu}}). Por quê: {{justificativa — ADR 0003}}.
- **Temperatura:** {{temperatura}} — extração pede a resposta mais provável e estável;
  4.4d testa {{temperatura_alt}}.

## 4. Auditoria: as quatro verificações (4.4)

- **a) Fidelidade:** {{n_fieis}}/{{n}} fichas com trecho literal no texto enviado;
  {{n_inventados}} trechos inventados (detectados por {{metodo_fidelidade}}).
- **b) Estabilidade:** {{n_identicas}}/{{n}} fichas idênticas entre rep1 e rep2;
  campo mais instável: {{campo_mais_instavel}} ({{taxa_campo}}).
- **c) Efeito da entrada:** {{n_discordancias_entrada}}/{{n}} fichas discordam entre
  {{estrategia_a}} e {{estrategia_b}}; principalmente em {{campos}}. Defendemos {{estrategia}}.
- **d) Efeito da temperatura:** em {{n_subconjunto}} artigos, t={{temperatura_alt}} muda
  {{mudancas_temp}}; a escolha de t={{temperatura}} {{se_sustenta}}.
- **Regra de confiança (declarada):** {{ConfidenceRule.describe()}}.
  Distribuição: alta {{n_alta}}, média {{n_media}}, baixa {{n_baixa}}.

## 5. Fichas que não defenderíamos

| arquivo | motivos |
|---|---|
| {{arquivo}} | {{motivos}} |

Destaque (o que "levanta" a nota): {{caso de invenção convincente, como foi detectado e por que
o método funcionou}}.

## 6. Custo (4.5)

- **Premissa de preço (visível):** USD {{preco_entrada}}/Mtok entrada, USD {{preco_saida}}/Mtok
  saída — {{fonte_preco}}.
- Tokens efetivamente enviados (todas as execuções, incluindo repetições):
  {{tokens_real}} → USD {{custo_real}}.
- Alternativa ingênua (artigos inteiros, mesmas execuções): {{tokens_ingenuo}} → USD {{custo_ingenuo}}.
- **Razão:** a ingênua é {{razao}}× mais cara.

## 7. O que faríamos diferente com mais uma semana

- {{item 1}}
- {{item 2}}
- {{item 3}}

## 8. Declaração de uso de IA

{{onde e como a IA foi usada; o grupo sabe explicar cada linha do código}}
