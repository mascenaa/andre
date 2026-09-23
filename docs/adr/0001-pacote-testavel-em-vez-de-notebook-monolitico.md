# ADR 0001 — Pacote testável em vez de notebook monolítico

**Status:** aceito · **Data:** 2026-09-23

## Contexto

O enunciado exige um notebook executado de ponta a ponta, e a rubrica valoriza decisões
rastreáveis, números legíveis e código que o grupo saiba explicar linha a linha.
Um notebook de 40 células com toda a lógica embutida é difícil de testar, de reexecutar
parcialmente e de auditar.

## Decisão

Toda a lógica fica em `src/ficha` (pacote instalável, tipado, com testes). O notebook
importa o pacote e faz apenas orquestração, exibição e narrativa. Um cliente de modelo
falso e determinístico (`ficha.llm.fake.FakeLLM`) permite rodar o pipeline inteiro —
inclusive as quatro verificações da auditoria — sem GPU, sem rede e sem os PDFs reais.

## Alternativas consideradas

- *Notebook único*: mais simples de entregar, mas cada mudança exige reexecutar tudo e
  nada é testável isoladamente.
- *Scripts soltos*: testáveis, mas sem fronteiras claras entre decisões (seleção vs prompt
  vs auditoria).

## Consequências

- O Colab precisa instalar o pacote (`pip install -e .` a partir de um zip ou do repositório).
- Ganhamos testes para cada decisão de projeto e um modo de ensaio que prova o pipeline
  antes de gastar GPU.
