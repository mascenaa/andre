# ruff: noqa: E501 — as strings abaixo são o código das células; a largura que importa é a
# do texto após ``textwrap.dedent`` (12 espaços a menos), não a do arquivo.
"""Gera o notebook de entrega com ``nbformat`` (ver ADR 0006).

Uso:
    python scripts/build_notebook.py
        → notebooks/AtividadeI_SOBRENOMES.ipynb
    python scripts/build_notebook.py --integrantes "Ana Souza" "João da Conceição"
        → notebooks/AtividadeI_Souza_Conceicao.ipynb

O notebook NÃO é editado à mão: mude este script e regenere (``make notebook``). As células de
código só orquestram o pacote ``ficha``; as de markdown narram as decisões (resumo dos ADRs).
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from ficha.report.naming import delivery_basename

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = "<<INTEGRANTES: Nome Sobrenome, Nome Sobrenome, ...>>"


def md(text: str) -> nbformat.NotebookNode:
    return new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str) -> nbformat.NotebookNode:
    return new_code_cell(textwrap.dedent(text).strip())


def cells(integrantes: list[str]) -> list[nbformat.NotebookNode]:
    nomes_md = "\n".join(f"- {n}" for n in integrantes) if integrantes else f"- {PLACEHOLDER}"
    nomes_py = repr(integrantes if integrantes else [PLACEHOLDER])
    return [
        # ------------------------------------------------------------------ 0. identificação
        md(
            f"""
            # Atividade de Construção I — A ficha que você teria de defender

            **Computação Cognitiva · Ciência de Dados e Negócios · ESPM · 2026.2 · Prof. André Insardi**

            **Integrantes**

            {nomes_md}

            > Preencha os nomes acima **e** a lista `INTEGRANTES` na célula de setup (ou regenere
            > com `python scripts/build_notebook.py --integrantes "Nome Sobrenome" ...`). O nome
            > dos três arquivos de entrega (`AtividadeI_<sobrenomes>`) é derivado dessa lista.

            **Declaração de uso de IA (Seção 7).** Usamos assistentes de IA para escrever partes do
            código do pacote `ficha` e revisar textos. Todas as decisões de projeto, os números e
            as conclusões foram verificados pelo grupo, que sabe explicar cada linha entregue.

            **Como ler este notebook.** Toda a lógica (limpeza, seleção, prompt, parse, auditoria,
            custo, relatório) vive no pacote testado `src/ficha`; aqui só orquestramos e
            mostramos resultados (ADR 0001). Há dois modos:

            - `MODO = "ensaio"`: `FakeLLM` determinístico + PDFs sintéticos gerados na hora —
              roda em segundos, sem GPU nem rede, e prova que o pipeline inteiro funciona.
              **Os números do modo ensaio não são resultados da atividade.**
            - `MODO = "real"`: os 19 PDFs em `data/raw` + Qwen2.5-3B-Instruct na T4 do Colab
              (ou o backend declarado). É este que gera a entrega.
            """
        ),
        # ------------------------------------------------------------------ 1. setup
        md(
            """
            ## 1. Setup

            **No Colab, antes de qualquer célula:** *Ambiente de execução → Alterar tipo → GPU T4*
            (a troca reinicia a sessão). Depois:

            1. suba `ficha.zip` (gerado por `make colab-zip`) para `/content/`;
            2. suba os 19 PDFs para `/content/ficha/data/raw/` (ou aponte `RAW_DIR` para o Drive);
            3. se for usar API, cadastre a chave nos **Secrets** do Colab (ícone de chave) com o nome
               `ANTHROPIC_API_KEY` — ela é lida para o ambiente **sem ser impressa**.

            Sementes fixas em `random`, `numpy` e `torch` (Seção 7, Reprodutibilidade).
            """
        ),
        code(
            f"""
            import importlib.util
            import os
            import shutil
            import subprocess
            import sys
            from pathlib import Path

            INTEGRANTES = {nomes_py}
            MODO = os.environ.get("FICHA_MODO", "ensaio")  # troque para "real" na entrega
            BACKEND_REAL = "qwen_local"  # ou "anthropic" / "openai_compat" (declare no relatório)
            RAW_DIR = None  # ex.: Path("/content/drive/MyDrive/artigos") — None = data/raw
            # Reutiliza execuções já gravadas em data/runs (mesma configuração) em vez de chamar
            # o modelo de novo — ver seção 6. FICHA_REUSAR=0 força reexecutar tudo.
            REUSAR_EXECUCOES = os.environ.get("FICHA_REUSAR", "1") == "1"

            try:
                import google.colab  # noqa: F401

                IN_COLAB = True
            except ImportError:
                IN_COLAB = False
            if IN_COLAB:
                ROOT = Path("/content/ficha")
                if not ROOT.exists() and Path("/content/ficha.zip").exists():
                    shutil.unpack_archive("/content/ficha.zip", ROOT)
                extras = "local,embeddings,api"
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", f"ficha[{{extras}}] @ file://{{ROOT}}"],
                    check=True,
                )
            else:
                cwd = Path.cwd()
                ROOT = Path(os.environ.get("FICHA_ROOT", cwd.parent if cwd.name == "notebooks" else cwd))
            ROOT.mkdir(parents=True, exist_ok=True)
            os.chdir(ROOT)

            if IN_COLAB:  # chaves: Secrets do Colab → ambiente, nunca impressas
                from google.colab import userdata

                for nome in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
                    try:
                        valor = userdata.get(nome)
                    except Exception:
                        valor = None
                    if valor:
                        os.environ[nome] = valor

            from ficha.extract import set_seeds
            from ficha.llm.qwen_local import describe_gpu

            set_seeds(42)
            print(f"MODO={{MODO}} · reusar execuções={{REUSAR_EXECUCOES}} · Colab={{IN_COLAB}} · raiz={{ROOT}}")
            print("GPU:", describe_gpu() or "nenhuma (ok no modo ensaio; no modo real use a T4)")
            print("Chaves no ambiente (só presença):",
                  {{n: bool(os.environ.get(n)) for n in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")}})
            """
        ),
        # ------------------------------------------------------------------ 2. configuração
        md(
            """
            ## 2. Configuração declarada

            Tudo o que o relatório precisa declarar vem de `ficha.config.Settings` (variáveis
            `FICHA_*`, `.env` ou Secrets): modelo, temperatura, semente, parâmetros de seleção e a
            **premissa de preço** usada na Seção 4.5. Nenhum segredo é impresso.
            """
        ),
        code(
            """
            import json

            import pandas as pd
            from IPython.display import Image, display

            from ficha.config import Settings, get_settings
            from ficha.llm import FakeBehavior, build_client

            pd.set_option("display.max_colwidth", 120)
            if MODO == "ensaio":
                DATA_DIR = ROOT / "data" / "outputs" / "ensaio"
                shutil.rmtree(DATA_DIR, ignore_errors=True)  # ensaio sempre do zero
                settings = get_settings(data_dir=DATA_DIR, model_backend="fake")
                # Erros simulados para a auditoria ter o que encontrar (ver ficha.llm.fake).
                client = build_client(settings, behavior=FakeBehavior(
                    broken_json_rate=0.25, invent_trecho_rate=0.2,
                    invent_limitacao_rate=0.4, unstable_rate=0.3,
                ))
            else:
                settings = get_settings(data_dir=ROOT / "data", model_backend=BACKEND_REAL)
                client = build_client(settings)
            settings.ensure_dirs()
            FIG_DIR = settings.outputs_dir / "figuras"
            FIG_DIR.mkdir(parents=True, exist_ok=True)
            count_tokens = client.count_tokens  # tokenizador real no Qwen; ~4 chars/token no fake

            print(json.dumps(settings.declaracao(), ensure_ascii=False, indent=2))
            print("cliente:", client.model_name)
            """
        ),
        # ------------------------------------------------------------------ 3. ingestão
        md(
            """
            ## 3. Ingestão e limpeza do texto

            `ficha.ingest` extrai o texto página a página com `pymupdf` e aplica uma limpeza
            determinística e conservadora (Seção 4.1, último parágrafo): remove cabeçalhos e
            rodapés repetidos e números de página soltos, desfaz a **hifenização de fim de linha**
            (com vocabulário do próprio artigo para não quebrar compostos legítimos como
            *state-of-the-art*), reconstrói parágrafos e normaliza espaços. O resultado fica em
            cache (`data/processed`, com `sha256`) para que todas as execuções usem o mesmo texto.
            """
        ),
        code(
            """
            from ficha.ingest import load_or_ingest
            from ficha.ingest.synthetic import generate_synthetic_corpus
            from ficha.report.overview import corpus_frame

            raw_dir = Path(RAW_DIR) if RAW_DIR else settings.raw_dir
            if MODO == "ensaio":
                artigos = generate_synthetic_corpus(raw_dir, n=6, seed=settings.seed)
                # Gabarito conhecido: quais artigos declaram limitação (null esperado nos demais).
                GABARITO_LIMITACAO = {a.arquivo: a.has_limitations for a in artigos}
            else:
                GABARITO_LIMITACAO = None  # sem gabarito: a auditoria usa a heurística declarada

            pdfs = sorted(raw_dir.glob("*.pdf"))
            assert pdfs, f"Nenhum PDF em {raw_dir}: suba os artigos antes de continuar."
            docs = [load_or_ingest(p, settings.processed_dir) for p in pdfs]
            corpus = corpus_frame(docs, count_tokens)
            display(corpus)
            total = corpus.iloc[-1]
            print(f"{len(docs)} artigos · {total.paginas} páginas · {total.caracteres:,} caracteres · "
                  f"~{total.tokens:,} tokens (enunciado: 19 artigos, ~476 páginas, ~1,4 M chars, ~350 k tokens)")
            meta = docs[0].metadata
            print({k: meta.get(k) for k in ("n_chars_raw", "n_chars_clean", "empty_pages")})
            """
        ),
        # ------------------------------------------------------------------ 4. seleção
        md(
            """
            ## 4. O que mandar para o modelo (Seção 4.1)

            Três estratégias implementadas, todas com a mesma interface (`ContextSelector`):

            | Estratégia | O que envia | Risco |
            |---|---|---|
            | `first_pages` | as N primeiras páginas | limitações e métricas costumam estar no fim |
            | `keyword` | seções achadas por título (*Methods*, *Results*, *Limitations*...) | depende do nome da seção |
            | `semantic` | blocos de ~1.200 caracteres (sobreposição de 200) mais próximos de consultas sobre problema, dados, método, métrica e limitação | depende da qualidade do embedding |

            **Escolha: `semantic`**, com `keyword` como alternativa (ADR 0002). Blocos de 1.200
            caracteres (~300 tokens, um parágrafo típico, a unidade em que autores declaram uma
            limitação) com sobreposição de 200 (uma frase inteira: a evidência não é cortada);
            8 consultas em português, uma ou duas por campo da ficha — incluindo "como o
            desempenho foi medido" e "o que este trabalho não consegue fazer" —, embedder
            multilíngue (as consultas são em português, os artigos em inglês) e blocos
            reordenados por página. A busca vetorial aqui **não é o produto** — é só o modo de
            escolher o que entra no prompt. A verificação 4.4c (seção 8)
            testa essa escolha contra `first_pages`. Abaixo: o que cada estratégia envia para um
            artigo, e o custo em tokens de cada uma no corpus inteiro.
            """
        ),
        code(
            """
            from ficha.report.overview import selection_frame, strategy_summary
            from ficha.select import SELECTOR_NAMES, HashingEmbedder, build_selector

            if MODO == "ensaio" or importlib.util.find_spec("sentence_transformers") is None:
                embedder = HashingEmbedder()  # offline, determinístico
                if MODO == "real":
                    print("ATENÇÃO: sentence-transformers ausente; usando HashingEmbedder (declare no relatório).")
            else:
                embedder = None  # SentenceTransformer multilíngue das Settings
            ESTRATEGIA = os.environ.get("FICHA_ESTRATEGIA", "semantic")  # principal (4.1)
            ESTRATEGIA_ALT = "first_pages"  # comparação 4.4c
            nomes = dict.fromkeys([*SELECTOR_NAMES, ESTRATEGIA, ESTRATEGIA_ALT])
            selectors = {nome: build_selector(nome, settings, embedder) for nome in nomes}

            exemplo = docs[0]
            for nome, sel in selectors.items():
                ctx = sel.select(exemplo)
                texto = ctx.render()
                print(f"=== {nome}: páginas {list(ctx.pages)} · {len(ctx.chunks)} trechos · "
                      f"{len(texto):,} chars · ~{count_tokens(texto):,} tokens · params={ctx.params}")
                print(texto[:500].rstrip(), "[...]\\n")
            """
        ),
        code(
            """
            selecao = selection_frame(docs, list(selectors.values()), count_tokens)
            resumo_estrategias = strategy_summary(selecao)
            display(resumo_estrategias)
            """
        ),
        md(
            """
            ### A limitação declarada chega ao modelo?

            O campo `limitacao` só pode ser preenchido se a frase em que os autores declaram a
            limitação estiver no texto enviado. `ficha.select.recall_table` mede isso sobre o
            **texto completo** de cada artigo: localiza as frases com declaração forte de
            limitação ("a limitation of this study", "is limited by", "beyond the scope of this
            work"...) e conta quantas chegam ao contexto de cada estratégia. É um diagnóstico
            offline — não chama o modelo.

            Se poucas chegam, o `null` em `limitacao` é sobretudo efeito da **seleção** (o modelo
            não pode declarar o que não recebe), e não da abstenção do modelo.
            """
        ),
        code(
            """
            import ficha.select as _select
            from ficha.report.assemble import coverage_sentence, final_pages_share

            recall_table = getattr(_select, "recall_table", None)
            if recall_table is None:
                COBERTURA_LIMITACAO = ""
                print("recall_table ainda não existe no pacote.")
            else:
                recall = recall_table(docs, selectors)
                display(recall)
                COBERTURA_LIMITACAO = coverage_sentence(recall, len(docs), final_pages_share(docs))
                if MODO == "real":  # revisão manual das frases detectadas (ADR 0002)
                    COBERTURA_LIMITACAO += (
                        " O detector é um regex ruidoso: na revisão manual só ~18 das frases são "
                        "limitações admitidas pelos autores (precisão ≈ 30%, ADR 0002)."
                    )
                print(COBERTURA_LIMITACAO)
                for d in docs[:4]:
                    paginas = sorted({p for p, _ in _select.limitation_sentences(d)})
                    print(f"  {d.arquivo[:30]}: frases de limitação nas páginas {paginas} de {d.n_pages}")
            """
        ),
        # ------------------------------------------------------------------ 5. prompt
        md(
            """
            ## 5. O prompt (Seção 4.2)

            Cada técnica é um bloco com cabeçalho fixo, ligável/desligável por `PromptFeatures`
            (ADR 0003): **papel** (`### PAPEL`, na mensagem de sistema), **instrução explícita** e
            **formato de saída** com o schema JSON embutido, **delimitadores**
            `<<<ARTIGO>>> ... <<<FIM_ARTIGO>>>` com o aviso de que nada ali dentro é ordem,
            **few-shot** com dois exemplos — um deles com `limitacao: null` — e **abstenção**
            ("melhor `null` do que inventado"). A cadeia de raciocínio existe como variante
            (`v_cot`, raciocínio num campo separado do JSON), mas não está na versão principal.

            **Medição obrigatória:** `v_full` × `v_sem_fewshot` diferem **só** no few-shot.

            **Temperatura 0,0.** Extração é tarefa de fidelidade: queremos a saída mais provável e
            reprodutível (com semente fixa). A seção 8d mede o que muda com 0,7.
            """
        ),
        code(
            """
            from ficha.prompts import build_prompt, describe_diff

            builder = build_prompt("v_full")
            rp = builder.render(selectors[ESTRATEGIA].select(exemplo))
            print("Blocos presentes:", builder.markers_present(), "· template_sha:", builder.template_sha())
            print("\\n--- SYSTEM ---\\n", rp.system)
            print("\\n--- USER (início) ---\\n", rp.user[:2200], "\\n[...]\\n", rp.user[-400:])
            print("\\n", describe_diff("v_full", "v_sem_fewshot"))
            print("\\nTemperatura:", settings.temperature, "—", Settings.model_fields["temperature"].description)
            """
        ),
        # ------------------------------------------------------------------ 6. extração
        md(
            """
            ## 6. Extração — matriz de execuções

            Cada execução tem um `run_id` e grava **as saídas brutas** do modelo em
            `data/runs/<run_id>/records.jsonl` (+ `manifest.json` com modelo, prompt, estratégia,
            temperatura e semente). A saída passa por `parse_completion`: JSON validado pelo
            schema (`FichaExtraida`, `extra="forbid"`); se vier malformado, tentamos reparos
            registrados (cercas de código, vírgula final, aspas tipográficas) e o status fica
            `repaired`; se nada funcionar, `failed` — o código trata e segue, não quebra.

            | Execução | Prompt | Estratégia | Temperatura | Artigos | Serve para |
            |---|---|---|---|---|---|
            | a) principal | v_full | semantic | 0,0 | todos | tabela final |
            | b) repetição | v_full | semantic | 0,0 | todos | 4.4b estabilidade |
            | c) prompt alternativo | v_sem_fewshot | semantic | 0,0 | todos | 4.2 comparação |
            | d) entrada alternativa | v_full | first_pages | 0,0 | todos | 4.4c efeito da entrada |
            | e) temperatura | v_full | semantic | 0,7 | subconjunto | 4.4d temperatura |

            **Reutilização (Seção 7, reprodutibilidade):** as saídas brutas gravadas são a fonte da
            auditoria; com `REUSAR_EXECUCOES`, uma execução com a mesma configuração (estratégia,
            variante, temperatura, repetição e artigos) é carregada de `data/runs` em vez de
            refeita — reexecutar custa ~1 h de GPU e não muda a auditoria com decodificação gulosa
            e seed fixa (a repetição da seção 8b mostra isso).
            """
        ),
        code(
            """
            from ficha.extract import ExtractionRunner, RunResult, RunStore, reparse
            from ficha.types import GenerationParams, ParseStatus

            store = RunStore(settings.runs_dir)

            def execucao_gravada(estrategia, variante, temperatura, repeticao, documentos):
                # run_id mais recente com a mesma configuração e os mesmos artigos, ou None.
                arquivos = sorted(d.arquivo for d in documentos)
                for run_id in reversed(store.list_runs()):
                    try:
                        m = store.manifest(run_id)
                    except FileNotFoundError:
                        continue
                    if (m.get("status_execucao") == "concluida"
                            and m["estrategia"]["nome"] == estrategia
                            and m["prompt"]["variante"] == variante
                            and float(m["temperatura"]) == float(temperatura)
                            and m["repeticao"] == repeticao
                            and m["n_docs"] == len(documentos)
                            and sorted(m.get("arquivos", arquivos)) == arquivos):
                        return run_id
                return None

            def rodar(estrategia, variante, temperatura, repeticao, documentos):
                if REUSAR_EXECUCOES:
                    run_id = execucao_gravada(estrategia, variante, temperatura, repeticao, documentos)
                    if run_id is not None:
                        res = RunResult(run_id, store.load(run_id), store.manifest(run_id))
                        print(f"reutilizado: {run_id}: {res.status_counts()}")
                        return res
                params = GenerationParams(temperature=temperatura, seed=settings.seed,
                                          max_new_tokens=settings.max_new_tokens)
                runner = ExtractionRunner(client, selectors[estrategia], build_prompt(variante), params, store)
                res = runner.run(documentos, repetition=repeticao, show_progress=(MODO == "real"))
                print(f"{res.run_id}: {res.status_counts()}")
                return res

            SUBCONJUNTO_T = docs[: max(3, len(docs) // 3)]  # 19 artigos → 6
            execucoes = {
                "principal": rodar(ESTRATEGIA, "v_full", settings.temperature, 1, docs),
                "repeticao": rodar(ESTRATEGIA, "v_full", settings.temperature, 2, docs),
                "prompt_alt": rodar(ESTRATEGIA, "v_sem_fewshot", settings.temperature, 1, docs),
                "entrada_alt": rodar(ESTRATEGIA_ALT, "v_full", settings.temperature, 1, docs),
                "temperatura_alt": rodar(ESTRATEGIA, "v_full", settings.temperature_alt, 1, SUBCONJUNTO_T),
            }
            display(pd.DataFrame([
                {"execucao": k, "run_id": r.run_id, **r.status_counts(),
                 "json_valido_de_primeira": r.manifest["resumo"]["taxa_json_valido_de_primeira"]}
                for k, r in execucoes.items()
            ]))
            """
        ),
        md(
            "Uma saída **bruta** do modelo, exatamente como foi gravada, e um caso que precisou de reparo ou falhou:"
        ),
        code(
            """
            registros = execucoes["principal"].records
            r0 = next((r for r in registros if r.parse_status == ParseStatus.OK), registros[0])
            print(f"[{r0.run_id} · {r0.arquivo} · {r0.parse_status.value}]\\n{r0.raw_output[:1500]}")

            problemas = [r for res in execucoes.values() for r in res.records if r.parse_status != ParseStatus.OK]
            if problemas:
                rp_ = problemas[0]
                print(f"\\n[{rp_.run_id} · {rp_.arquivo} · {rp_.parse_status.value}] "
                      f"reparos={reparse(rp_).repairs} erro={rp_.error}\\n{rp_.raw_output[:1200]}")
            else:
                print("\\nNenhuma saída precisou de reparo nesta rodada — isso também é resultado.")
            """
        ),
        # ------------------------------------------------------------------ 7. prompts
        md(
            """
            ## 7. Comparação das duas versões do prompt (Seção 4.2)

            Mesmos artigos, mesma estratégia, mesma temperatura; só o few-shot muda. Medimos a taxa
            de **JSON válido de primeira**, a taxa de **`null` quando a informação não existe**
            (no ensaio contra o gabarito dos PDFs sintéticos; no real pela heurística de
            vocabulário declarada em `ficha.audit.runstats`), a fidelidade e **em que campos as
            fichas discordam**. A versão final sai da regra declarada
            (`PromptComparison.decide`: JSON válido → fidelidade → null correto), não da intuição.
            """
        ),
        code(
            """
            from ficha.audit import compare_prompt_variants
            from ficha.report.figures import plot_prompt_comparison

            comparacao = compare_prompt_variants(
                execucoes["principal"].records, execucoes["prompt_alt"].records,
                limitacao_gabarito=GABARITO_LIMITACAO,
            )
            tabela_prompts = comparacao.to_frame()
            display(tabela_prompts)
            veredito = comparacao.decide()
            print("Vencedora:", veredito.winner, "—", veredito.explanation)
            divergencias = pd.DataFrame([d.to_dict() for d in comparacao.disagreements()])
            display(divergencias.head(10) if not divergencias.empty else "As fichas das duas versões não discordam.")

            taxas = tabela_prompts.loc[[i for i in tabela_prompts.index if str(i).startswith("rate_")],
                                       list(comparacao.variants)].astype(float)
            fig_prompts = plot_prompt_comparison(taxas, FIG_DIR / "comparacao_prompts.png")
            display(Image(filename=str(fig_prompts)))
            # Versão compacta (só as taxas decisivas) para caber no relatório de 3 páginas.
            chave = ["rate_json_valid_first_try", "rate_null_when_expected", "rate_fidelity"]
            fig_prompts_relatorio = plot_prompt_comparison(
                taxas.loc[[k for k in chave if k in taxas.index]], FIG_DIR / "comparacao_prompts_relatorio.png"
            )
            """
        ),
        # ------------------------------------------------------------------ 8. auditoria
        md(
            """
            ## 8. Auditoria (Seção 4.4)

            As quatro verificações, cada uma com números e uma frase de conclusão (ADR 0005).

            - **a) Fidelidade** — `evidencia.trecho` é procurado no **texto que foi enviado** ao
              modelo (não no PDF inteiro): substring exata após normalização tipográfica; senão
              `partial_ratio` ≥ 0,90. Também conferimos se a página declarada bate com o marcador
              `[p. N]` onde o trecho está.
            - **b) Estabilidade** — execução repetida sem mudar nada; diferença campo a campo.
            - **c) Efeito da entrada** — `semantic` × `first_pages` nos mesmos artigos.
            - **d) Efeito da temperatura** — 0,0 × 0,7 num subconjunto.
            """
        ),
        code(
            """
            from ficha.audit import RULE_V1, RULE_V2, build_audit_summary
            from ficha.report.assemble import fidelity_block, input_block, stability_block, temperature_block

            resumo = build_audit_summary(
                execucoes["principal"].records,
                stability_rep=execucoes["repeticao"].records,
                alt_input=execucoes["entrada_alt"].records,
                t_alt=execucoes["temperatura_alt"].records,
                prompt_runs=(execucoes["principal"].records, execucoes["prompt_alt"].records),
                limitacao_gabarito=GABARITO_LIMITACAO,
                rule=RULE_V2,  # regra final — ver a calibração na seção 9
            )
            tabelas = resumo.tables()

            def mostrar(bloco):
                print(f"{bloco.titulo}: " + " · ".join(f"{k}: {v}" for k, v in bloco.numeros.items()))
                print("→", bloco.conclusao)

            mostrar(fidelity_block(resumo))
            display(tabelas["fidelidade"])
            """
        ),
        md(
            "Como a fidelidade pega um trecho inventado: o trecho devolvido, o score e o que o texto enviado realmente tinha."
        ),
        code(
            """
            principal = {r.arquivo: r for r in execucoes["principal"].records}
            falhas = resumo.fidelity.failures()
            for f in falhas[:3]:
                rec = principal[f.arquivo]
                trecho = rec.ficha.evidencia.trecho if rec.ficha else "(sem ficha)"
                print(f"{f.arquivo}: score={f.score:.2f} ({f.method}) — trecho devolvido:\\n  «{trecho}»")
                print(f"  início do texto enviado: «{rec.context_text[:300]}...»\\n")
            if not falhas:
                print("Todos os trechos foram encontrados no texto enviado.")
            """
        ),
        code(
            """
            from ficha.report.figures import plot_field_change_rates

            mostrar(stability_block(resumo))
            display(tabelas["estabilidade_campos"])
            fig_estab = plot_field_change_rates(
                tabelas["estabilidade_campos"][["field", "change_rate"]], FIG_DIR / "estabilidade_campos.png",
                title="4.4b — taxa de mudança por campo entre repetições",
            )
            display(Image(filename=str(fig_estab)))
            """
        ),
        code(
            """
            mostrar(input_block(resumo))
            display(tabelas["entrada_estrategias"])
            display(tabelas["entrada_campos"])
            display(tabelas["entrada_divergencias"].head(10))
            fig_entrada = plot_field_change_rates(
                tabelas["entrada_campos"][["field", "change_rate"]], FIG_DIR / "entrada_campos.png",
                title=f"4.4c — campos que mudam entre {ESTRATEGIA} e {ESTRATEGIA_ALT}",
            )
            display(Image(filename=str(fig_entrada)))
            """
        ),
        code(
            """
            mostrar(temperature_block(resumo))
            print("Critério declarado:", resumo.temperature.criterio)
            display(tabelas["temperatura"])
            """
        ),
        # ------------------------------------------------------------------ 9. confiança
        md(
            """
            ## 9. Confiança — regra declarada e fichas que não defenderíamos

            `confianca` nunca vem do modelo nem "do olho": é função determinística das verificações
            acima (`ficha.audit.ConfidenceRule`). Cada ficha carrega os **motivos** que a
            impediram de ter nível mais alto — é deles que sai a lista de fichas não defensáveis.

            **Duas versões da regra, e por que a final é a v2.** A v1 (estrita) rebaixava uma ficha
            se *qualquer* campo mudasse entre as duas estratégias de entrada. Os dados mostram que
            isso mede redação, não confiabilidade: entre estratégias, a similaridade média dos
            campos de texto livre fica em torno de 0,5 (entradas diferentes produzem paráfrases e
            evidências de outras páginas por construção), enquanto entre repetições com a mesma
            entrada é 1,00. A v2 compara estratégias só pela **discordância categórica** — a
            limitação declarada numa e `null` na outra — e mantém a comparação exata entre
            repetições. A tabela abaixo mostra as duas distribuições lado a lado; a tabela final e
            o PDF usam a v2. A calibração foi decidida depois de ver os dados reais, e isso é
            declarado aqui e no relatório.
            """
        ),
        code(
            """
            from ficha.audit import build_final_fichas, confidence_distribution
            from ficha.report.assemble import mean_text_similarity

            args_regra = (execucoes["principal"].records, execucoes["repeticao"].records,
                          execucoes["entrada_alt"].records)
            DISTRIBUICOES = {
                "v1 estrita": confidence_distribution(build_final_fichas(*args_regra, RULE_V1)),
                "v2 categórica (final)": confidence_distribution(build_final_fichas(*args_regra, RULE_V2)),
            }
            display(pd.DataFrame(DISTRIBUICOES).rename_axis("confianca"))
            sim_entrada = mean_text_similarity(resumo.input_effect.diff.field_mean_similarity)
            sim_repeticao = mean_text_similarity(resumo.stability.diff.field_mean_similarity)
            print(f"Similaridade média dos campos de texto livre: entre estratégias {sim_entrada:.2f} · "
                  f"entre repetições {sim_repeticao:.2f}")
            """
        ),
        code(
            """
            from ficha.report.figures import plot_confidence_distribution

            print(resumo.rule.describe())
            display(tabelas["confianca"])
            fichas_finais = [f.ficha for f in resumo.fichas]
            fig_conf = plot_confidence_distribution(fichas_finais, FIG_DIR / "distribuicao_confianca.png")
            display(Image(filename=str(fig_conf)))
            print("Fichas que NÃO defenderíamos (confiança baixa) e por quê:")
            display(tabelas["nao_defensaveis"])
            """
        ),
        code(
            """
            colunas = ["arquivo", "confianca", "limitacao", "evidencia_pagina", "audit_motivos",
                       "audit_fidelity_score", "audit_campos_instaveis", "audit_campos_divergentes_entrada"]
            display(tabelas["fichas"][colunas])
            """
        ),
        # ------------------------------------------------------------------ 10. custo
        md(
            """
            ## 10. Custo (Seção 4.5)

            Somamos o `Usage` de **todas** as chamadas feitas (as cinco execuções, inclusive
            repetição, prompt alternativo, entrada alternativa e temperatura). A alternativa
            ingênua espelha exatamente as mesmas chamadas, trocando o texto selecionado pelo
            artigo inteiro. Mesmo com o modelo local gratuito, a premissa de preço (visível abaixo)
            responde "quanto isso custaria" num provedor pago.
            """
        ),
        code(
            """
            from ficha.cost import (PricePremise, compare_costs, cost_by_run, cost_of_records,
                                    cost_table, naive_cost_from_records)

            premissa = PricePremise.from_settings(settings)
            todos = [r for res in execucoes.values() for r in res.records]
            custo_real = cost_of_records(todos, premissa, label="real (todas as execuções)")
            custo_ingenuo = naive_cost_from_records(todos, docs, count_tokens, premissa)
            comparacao_custo = compare_costs(custo_real, custo_ingenuo)
            custo_por_execucao = cost_by_run(todos, premissa)
            print(premissa.describe())
            display(cost_table([*custo_por_execucao, custo_real, custo_ingenuo]))
            print(comparacao_custo.describe())
            """
        ),
        # ------------------------------------------------------------------ 11. exportação
        md(
            """
            ## 11. Tabela comparativa (entregável 2)

            Uma linha por artigo, com **todos** os campos da ficha na ordem da Seção 3, mais a
            coluna `motivos_confianca`. CSV em UTF-8 com BOM (abre certo no Excel pt-BR) e XLSX
            formatado. `limitacao = null` vira célula vazia.
            """
        ),
        code(
            """
            from ficha.report import delivery_basename, write_table

            BASENAME = delivery_basename(INTEGRANTES)
            extras = {f.arquivo: {"motivos_confianca": f.motivos} for f in resumo.fichas}
            saidas = write_table(fichas_finais, settings.outputs_dir, basename=BASENAME, extras=extras)
            print("CSV :", saidas.csv)
            print("XLSX:", saidas.xlsx)
            display(pd.read_csv(saidas.csv, encoding="utf-8-sig").head())
            """
        ),
        # ------------------------------------------------------------------ 12. relatório
        md(
            """
            ## 12. Relatório PDF (entregável 3, ≤ 3 páginas)

            O relatório é montado **a partir dos objetos acima** (`ficha.report.assemble`): nenhum
            número é digitado à mão. Os textos de justificativa vêm de `Narrative` (resumo dos
            ADRs); depois de ler os resultados reais, o grupo revisa e sobrescreve o que quiser —
            inclusive as conclusões de cada verificação (`Narrative(conclusoes={...})`). A geração
            conta as páginas e falha alto se passar de 3.
            """
        ),
        code(
            """
            from ficha.report import build_report_pdf, count_pages
            from ficha.report.assemble import Narrative, build_report_content
            from ficha.report.pdf import ReportTooLongError

            narrativa = Narrative(  # edite aqui; conclusoes={"fidelidade": "..."} sobrescreve blocos
                cobertura_limitacao=COBERTURA_LIMITACAO,
                nota_limitacao=(
                    "Sem gabarito manual não dá para separar abstenção correta de omissão; a "
                    "heurística de vocabulário de limitação teve falsos positivos confirmados "
                    "(título 'Scope, and Limitations'; parágrafo sobre limitações da IA "
                    "generativa) e por isso não entra na regra."
                ) if MODO == "real" else "",
            )
            gpu = describe_gpu()
            MODELO_DECLARADO = client.model_name + (f" · {gpu['nome']}" if gpu else "")
            if MODO == "ensaio":
                MODELO_DECLARADO += " (MODO ENSAIO: FakeLLM + PDFs sintéticos — não é resultado)"
            conteudo = build_report_content(
                integrantes=INTEGRANTES,
                modelo=MODELO_DECLARADO,
                summary=resumo,
                cost=comparacao_custo,
                cost_per_run=custo_por_execucao,
                strategy_table=resumo_estrategias[["caracteres_medios", "tokens_medios", "tokens_corpus"]],
                narrative=narrativa,
                figuras=[fig_prompts_relatorio],
                rule_distributions=DISTRIBUICOES,
            )
            pdf_path = settings.outputs_dir / f"{BASENAME}.pdf"
            try:
                build_report_pdf(conteudo, pdf_path)
                print(f"PDF: {pdf_path} · {count_pages(pdf_path)} página(s)")
            except ReportTooLongError as exc:
                print("ATENÇÃO — relatório longo demais, encurte a narrativa:", exc)
            """
        ),
        # ------------------------------------------------------------------ 13. checklist
        md(
            """
            ## 13. Checklist de entrega

            - [ ] `MODO = "real"` e os 19 PDFs em `data/raw` (a tabela da seção 3 mostra 19 linhas).
            - [ ] Integrantes preenchidos na **primeira célula** e em `INTEGRANTES`; o PDF traz os
                  nomes na primeira página.
            - [ ] Os três arquivos nomeados `AtividadeI_<sobrenomes>`: este notebook (renomeie o
                  `.ipynb`), `data/outputs/AtividadeI_<sobrenomes>.csv`/`.xlsx` e `.pdf`.
            - [ ] **Ambiente de execução → Reiniciar e executar tudo**, sem erro, com as saídas
                  visíveis — só então baixar o `.ipynb`.
            - [ ] O PDF tem no máximo 3 páginas (a seção 12 imprime o número).
            - [ ] Nenhuma chave no notebook: só Secrets do Colab/variáveis de ambiente (a célula de
                  setup imprime apenas se a chave existe).
            - [ ] As conclusões do relatório foram revisadas pelo grupo em `Narrative(...)` e cada
                  frase é defensável com um número deste notebook.
            - [ ] As saídas brutas (`data/runs/`) foram guardadas.
            """
        ),
    ]


def build(integrantes: list[str], out: Path | None = None) -> Path:
    nb = new_notebook(cells=cells(integrantes))
    nb.metadata.update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            # Abre no Colab já pedindo GPU T4 (Seção 4.3).
            "accelerator": "GPU",
            "colab": {"gpuType": "T4", "provenance": []},
        }
    )
    out = out or ROOT / "notebooks" / f"{delivery_basename(integrantes)}.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    nbformat.validate(nb)
    nbformat.write(nb, out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--integrantes", nargs="*", default=[], help="Nomes completos.")
    parser.add_argument("--out", type=Path, default=None, help="Caminho do .ipynb.")
    args = parser.parse_args(argv)
    path = build(args.integrantes, args.out)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
