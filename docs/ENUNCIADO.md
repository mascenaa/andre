# Enunciado — Atividade de Construção I (texto extraído do PDF oficial)
> Fonte: `Atividade_Construcao_I_Ficha_Comparativa_v2-1.pdf`, Prof. André Insardi, ESPM 2026.2.


---

<!-- página 1 -->

Computação Cognitiva · Atividade de Construção I
ESPM · 2026.2
Atividade de Construção I
A ficha que você teria de defender
Computação Cognitiva · Ciência de Dados e Negócios · ESPM · 2026.2 Prof. André Insardi
Formato
Em grupo, de 3 a 4 integrantes
Entrega
Domingo, 27/09/2026, às 23h59, pelo Canvas
Peso
Ver a descrição da atividade no Canvas
1. A situação
Você entrou num grupo de pesquisa na segunda-feira. Na sexta, o coordenador quer uma tabela comparativa
dos 19 artigos da bibliografia: o que cada trabalho tenta resolver, com que dados, por qual método, medido
com qual métrica, e qual limitação os próprios autores admitem.
São 476 páginas. Ler tudo em quatro dias não dá.
Usar um modelo de linguagem para preencher a tabela é fácil — em duas horas você tem 19 linhas bem escritas.
O problema é que algumas estarão erradas, e você não terá como saber quais.
E é a sua assinatura que vai na tabela. Se o coordenador apontar uma linha e perguntar “de onde saiu isso?”,
a resposta precisa existir.
O trabalho não é produzir a tabela. É produzir a tabela e provar que ela é confiável. Um grupo que
entregar 19 fichas impecáveis sem nenhuma verificação tira menos do que um grupo que entregar 19
fichas e mostrar, com evidência, que 4 delas não são confiáveis e por quê.
2. O material
Os 19 artigos em PDF estão no Canvas, no material da disciplina. Todos em inglês. Cerca de 476 páginas e 1,4
milhão de caracteres — algo em torno de 350 mil tokens.
O material é o mesmo para todos os grupos, de propósito: qualquer diferença nas conclusões terá vindo das
suas decisões, não de terem usado bases diferentes.
3. A ficha
Para cada um dos 19 artigos, exatamente estes campos:
{
"arquivo": "string",
"problema": "string — o que o trabalho tenta resolver, em uma frase",
"dados": "string — que dados usa (fonte, período, volume, se informado)",
"metodo": "string — a abordagem principal",
"metrica": "string — como o resultado é medido",
1


---

<!-- página 2 -->

Computação Cognitiva · Atividade de Construção I
ESPM · 2026.2
"limitacao": "string ou null — limitação que os AUTORES declaram",
"evidencia": {
"trecho": "string — o texto do artigo que sustenta os campos acima",
"pagina": "inteiro"
},
"confianca": "alta | media | baixa"
}
Dois campos merecem atenção.
limitacao deve trazer o que os autores reconhecem, não o que você acha do trabalho. Se o artigo não
declarar nenhuma, o valor correto é null. Preencher com uma limitação plausível inventada pelo modelo é o
erro mais comum e o mais grave desta atividade.
evidencia é o que transforma a ficha em algo defensável. Sem o trecho de origem e a página, a linha da
tabela é apenas uma afirmação.
4. As cinco frentes
4.1 Decidir o que mandar para o modelo
Um artigo de 40 páginas não cabe no prompt — e, mesmo que coubesse, seria caro e pioraria o resultado.
Então a primeira decisão de projeto é: que pedaço de cada artigo você envia?
Há caminhos diferentes, e você precisa escolher um e justificar:
• as primeiras páginas, assumindo que resumo e introdução bastam;
• as seções encontradas por palavra-chave (“Methods”, “Results”, “Limitations”);
• os trechos localizados por similaridade semântica, buscando dentro do próprio artigo os pedaços que
mais se aproximam de ideias como “como o desempenho foi medido” ou “o que este trabalho não con-
segue fazer”.
Note que o terceiro caminho usa busca vetorial — mas aqui ela não é o produto. É só o modo de escolher o
que entra no prompt. Se você segmentar os artigos, as decisões de tamanho e sobreposição continuam sendo
suas, e continuam precisando de justificativa.
Antes de tudo isso, o texto precisa sair do PDF em condições de uso: hifenização de fim de linha, quebras de
parágrafo e espaçamento irregular atrapalham tanto a busca quanto o prompt.
4.2 Escrever o prompt — e provar que ele é bom
Esta frente não é “peça a ficha ao modelo”. É aplicar as técnicas que vimos e medir qual delas muda alguma
coisa.
O seu prompt de extração precisa usar, de forma identificável:
• instrução explícita do que fazer e do formato exato de saída — nada de deixar implícito;
• papel, situando o modelo na tarefa;
• delimitadores separando com clareza a instrução do texto do artigo. Esse cuidado não é estético: sem
ele, uma frase do próprio artigo pode ser lida como se fosse ordem sua;
• few-shot — pelo menos um exemplo de ficha preenchida corretamente, incluindo um caso em que o
campo correto é null;
2


---

<!-- página 3 -->

Computação Cognitiva · Atividade de Construção I
ESPM · 2026.2
• instrução de abstenção: o modelo deve devolver null em vez de preencher com algo plausível. Veri-
fique se ele obedece.
Sobre cadeia de raciocínio: você pode pedir que o modelo raciocine antes de responder, mas lembre que isso
conflita com saída estruturada — a resposta deixa de ser JSON puro. Se usar, resolva o conflito (por exemplo,
separando raciocínio e resposta em campos distintos) e comente se valeu a pena.
A medição obrigatória. Escreva duas versões do prompt que difiram em uma técnica — por exemplo, com e
sem few-shot, ou com e sem delimitadores — rode as duas sobre o mesmo conjunto de artigos e compare:
• a taxa de saídas que são JSON válido de primeira;
• a taxa de campos preenchidos com null quando a informação realmente não existe;
• se as fichas resultantes discordam, e em quê.
Uma conclusão do tipo “a técnica X não mudou nada neste caso” é perfeitamente aceitável — desde que
sustentada pelos números. O que não se aceita é escolher a versão final por intuição.
Declare também a temperatura e justifique a escolha para uma tarefa de extração.
A saída precisa ser JSON válido e validado por esquema. Se vier malformado, seu código detecta e trata; não
quebra.
4.3 Qual modelo usar
Você não precisa de chave de API paga. Há dois caminhos, e ambos são aceitos:
Modelo local, na GPU gratuita do Colab. Use a família Qwen2.5-Instruct, que roda bem na T4 do plano
gratuito. Ative a GPU em Ambiente de execução →Alterar tipo →GPU T4 antes de rodar qualquer célula,
porque a troca reinicia a sessão.
Modelo
Cabe na T4?
Observação
Qwen2.5-1.5B-Instruct
com folga
o padrão dos nossos
laboratórios; rápido, mas erra
bastante nesta tarefa
Qwen2.5-3B-Instruct
sim, em meia precisão
o melhor equilíbrio para esta
atividade
Qwen2.5-7B-Instruct
só quantizado em 4 bits
mais capaz, mais lento, e exige
carregar quantizado
A T4 gratuita tem cerca de 16 GB de memória de vídeo. Em meia precisão, a regra de bolso é 2 GB por bilhão
de parâmetros, e ainda sobra o espaço do contexto — por isso o 7B em precisão cheia não cabe.
Modelo por API, se o grupo tiver acesso. Declare qual usou.
Um modelo pequeno errar mais não é problema desta atividade — é matéria-prima dela. Quanto mais
o modelo falha, mais a sua auditoria da Seção 4.4 tem o que mostrar. Um grupo que rodar um modelo
local, medir a taxa de JSON malformado e as invenções, e explicar o padrão dos erros, entrega um trabalho
melhor do que um grupo que usou um modelo grande e não verificou nada.
Se o grupo tiver acesso aos dois, comparar modelo local e modelo por API nas mesmas fichas é um excelente
acréscimo — e conta no critério de julgamento.
3


---

<!-- página 4 -->

Computação Cognitiva · Atividade de Construção I
ESPM · 2026.2
4.4 Auditar — a parte que vale mais
Esta é a frente que separa os trabalhos. Faça, no mínimo, as quatro verificações abaixo.
a) Fidelidade. O campo evidencia.trecho aparece mesmo no artigo? Verifique automaticamente, com-
parando o trecho devolvido com o texto que você enviou. Quantas das 19 fichas passam? Um trecho que não
existe no artigo é um trecho inventado.
b) Estabilidade. Rode a extração duas vezes para os mesmos artigos, sem mudar nada. As fichas são iguais?
Quais campos mudam mais? Quantifique a divergência — não basta dizer que “variou um pouco”.
c) Efeito da entrada. Extraia os mesmos artigos a partir de duas estratégias diferentes da Seção 4.1 — por
exemplo, só as primeiras páginas contra trechos selecionados por similaridade. Onde as fichas discordam?
Qual estratégia você defenderia, e com base em quê?
d) Efeito da temperatura. Repita um subconjunto com temperatura diferente e relate o que muda. Sua escolha
da Seção 4.2 se sustenta?
Ao final, o campo confianca de cada ficha deve refletir o que essas verificações mostraram — e você precisa
declarar a regra que usou para atribuí-lo. confianca preenchido “no olho” não conta.
4.5 Custo
Com contas explícitas e uma premissa de preço declarada e visível junto do resultado:
• quantos tokens você efetivamente enviou, somando os 19 artigos;
• quanto isso custaria;
• quanto custaria a alternativa ingênua de mandar os artigos inteiros;
• quantas vezes uma é mais cara que a outra.
Se você rodar a extração mais de uma vez (e vai, por causa de 4.4b), inclua isso na conta.
5. O que entregar
Três arquivos, enviados por um integrante do grupo:
1. O notebook (.ipynb), executado, com as saídas visíveis. Notebook sem saída, ou que não roda de ponta
a ponta, não é corrigido. Antes de enviar, use Reiniciar e executar tudo.
2. A tabela comparativa dos 19 artigos, em CSV ou XLSX, uma linha por artigo, com todos os campos da ficha.
3. Um relatório em PDF, de no máximo 3 páginas, contendo:
• a estratégia escolhida em 4.1 e por quê;
• as duas versões do prompt, o que muda entre elas e o que a comparação mostrou;
• qual modelo você usou e por quê;
• os resultados das quatro verificações de 4.4, com números;
• quais fichas você não defenderia e o motivo;
• o custo de 4.5;
• o que você faria diferente com mais uma semana.
Identifique todos os integrantes na primeira célula e na primeira página. Nomeie os arquivos como Ativi-
dadeI_<sobrenomes>.
4


---

<!-- página 5 -->

Computação Cognitiva · Atividade de Construção I
ESPM · 2026.2
6. Rubrica
Critério
Peso
O que é avaliado
Preparação e escolha do que
enviar
10 %
O texto foi tratado; a estratégia
da Seção 4.1 foi escolhida e
justificada, não adotada por
acaso
Engenharia de prompt
20 %
As técnicas exigidas aparecem
de forma identificável; as duas
versões do prompt foram
comparadas com números, e a
escolha final decorre da
medição
Extração
10 %
JSON validado por esquema;
ausência tratada como null;
temperatura e modelo
declarados e justificados
Auditoria
30 %
As quatro verificações foram
feitas e quantificadas; o campo
confianca segue uma regra
declarada
Custo
10 %
Contas explícitas, premissa
visível, comparação com a
alternativa ingênua
Julgamento
15 %
O relatório diz quais fichas não
sustentaria e por quê — com
honestidade e com causa
Clareza
5 %
Decisões rastreáveis, números
legíveis
O que derruba a nota: entregar 19 fichas sem nenhuma verificação; dizer que “o modelo acertou” sem medir;
confianca atribuído por impressão; escolher o prompt final por intuição.
O que levanta: encontrar uma ficha em que o modelo inventou de forma convincente, mostrar como você
detectou, e explicar por que o método de detecção funcionou naquele caso.
7. Regras
Uso de IA. Permitido e esperado — a disciplina é sobre isso. Mas declare onde usou e esteja preparado para
explicar qualquer linha do seu código. Código que o grupo não sabe explicar é tratado como não entregue.
Nada de framework pronto de extração ou de RAG. Bibliotecas de leitura de PDF, de embeddings e de banco
vetorial são permitidas e esperadas. O prompt, a validação e a auditoria são seus.
5


---

<!-- página 6 -->

Computação Cognitiva · Atividade de Construção I
ESPM · 2026.2
Chaves de API nunca no código. Variável de ambiente ou os Secrets do Colab. Chave exposta no notebook
entregue é falha de segurança e será descontada.
Reprodutibilidade. Fixe a semente onde houver aleatoriedade. Guarde as saídas brutas do modelo — sem
elas você não consegue fazer a verificação de estabilidade.
Atraso. A entrega fecha no horário. Problema de última hora com o Colab não é justificativa.
8. Perguntas frequentes
Precisa de GPU? Depende do caminho. Com modelo por API, não. Com modelo local, sim — use a T4 gratuita
do Colab, e troque o tipo de ambiente antes de rodar qualquer célula, porque a troca reinicia a sessão. A
etapa de embeddings, se você usá-la, roda em CPU em alguns minutos.
Isso não é o mesmo laboratório que fizemos em aula? Não. Lá o produto era um buscador, e a pergunta era
se ele encontrava. Aqui o produto é um conjunto de afirmações sobre 19 artigos, e a pergunta é se dá para
confiar nelas. A busca, se você usá-la, é apenas um meio de escolher o que enviar ao modelo.
E se o modelo acertar tudo? Improvável, e é justamente o que você tem de verificar em vez de supor. Se
acertar, mostre a evidência de que acertou — isso também é resultado.
Preciso pagar por uma API? Não. A Seção 4.3 mostra como rodar o Qwen2.5 na GPU gratuita do Colab. Declare
o que usou, seja qual for a escolha.
O modelo local está devolvendo JSON quebrado o tempo todo. Isso é um resultado, não um impedimento.
Meça a taxa, relate, e tente melhorar pelo prompt — é exatamente o que a Seção 4.2 pede. Se ainda assim
inviabilizar a tarefa, suba de tamanho de modelo e relate a diferença.
Computação Cognitiva · ESPM · Prof. André Insardi · 2026.2
6
