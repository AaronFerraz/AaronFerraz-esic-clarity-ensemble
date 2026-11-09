# EP1 — Classificação de Clareza em Respostas e-SIC

Projeto acadêmico de Processamento de Linguagem Natural, desenvolvido no contexto do EP1 da USP. Compara TF-IDF com classificadores tradicionais e fine-tuning de BERTimbau Base para predizer avaliações atribuídas por cidadãos a respostas de serviços públicos.

## Tarefa

Entrada: `resp_text`. Alvo: `clarity`.

| Classe | Nota original |
|---|---|
| `c1` | 1 |
| `c234` | 2, 3 ou 4 |
| `c5` | 5 |

O alvo é uma avaliação subjetiva. O texto da resposta não representa necessariamente a pergunta original, a expectativa do cidadão ou todo o contexto do atendimento.

## Métodos

- Baseline: TF-IDF de palavras e regressão logística multiclasse.
- Seleção clássica: regressão logística, LinearSVC, ComplementNB e SGDClassifier com modified Huber.
- Representações clássicas: TF-IDF de palavras, ou palavras e caracteres combinados por FeatureUnion.
- Rede pré-treinada: `neuralmind/bert-base-portuguese-cased`.
- Seleção: GridSearchCV com cinco dobras StratifiedGroupKFold.

Os grupos são construídos com normalização NFKC, conversão para minúsculas e normalização de espaços. Textos equivalentes por essa regra ficam na mesma dobra. Transformações aprendidas dos textos ficam dentro dos pipelines.

O BERTimbau processa textos longos em janelas sobrepostas de até 512 tokens. Cada janela herda o rótulo do documento no treino; na previsão, a média dos logits das janelas produz uma classe por documento. Assim, documentos longos recebem maior peso efetivo no treinamento.

## Estrutura

```text
EP1/
├── baseline_classifier.py
├── classical_model_selection.py
├── bertimbau_classifier.py
├── data_utils.py
├── analise_resultados.ipynb
├── requirements.txt
├── Dataset/
├── Report/
├── baseline_results/
├── classical_selection/
└── modelo_bertimbau/
```

`data_utils.py` centraliza a validação de planilhas, rótulos e caminhos. Os scripts preservam as planilhas de origem e exigem a coluna `clarity` vazia no teste de entrega.

## Resultados históricos disponíveis

| Método | Acurácia média na CV de seleção |
|---|---:|
| Baseline TF-IDF + LR | 45,39% |
| Melhor clássico: TF-IDF de palavras + LR | 45,71% |
| BERTimbau Base | 45,68% |

As demais alternativas clássicas e os parâmetros registrados estão no notebook e nos arquivos JSON de resumo.

**Essas são métricas da validação utilizada para escolher hiperparâmetros, e não uma avaliação independente de teste.** O nome histórico `oof_accuracy` da baseline não significa que um vetor de predições OOF foi armazenado. Novas execuções usam a chave `selection_cv_accuracy`.

Os resumos históricos foram preservados. Os grids presentes no código podem ter sido ampliados depois dos experimentos registrados; por isso, os resumos não comprovam a execução de todas as configurações atuais.

A diferença entre os melhores valores é pequena e não demonstra superioridade estatística. O Teste 1 disponível nesta pasta não tem referência rotulada, impossibilitando calcular sua acurácia e uma matriz de confusão a partir desses arquivos.

## Ambiente e instalação

Recomendação: Python 3.12. Os scripts tradicionais podem ser executados em CPU. O BERTimbau usa CUDA quando disponível e CPU caso contrário, com custo de execução consideravelmente maior.

Na raiz do projeto, no Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

Para usar GPU, instale primeiro a distribuição de PyTorch adequada ao sistema e driver conforme https://pytorch.org/get-started/locally/ . Depois:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

O requirements descreve faixas de compatibilidade e não é um lockfile exato. Após validar um ambiente, registre-o com:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > requirements-lock.txt
```

A revisão não instalou pacotes nem substituiu o ambiente existente. Não há garantia de reprodução numérica exata entre versões, plataformas e dispositivos.

## Dados

Disponha os arquivos autorizados em:

```text
Dataset/train.xlsx
Dataset/test1.xlsx
```

Ambos precisam conter `resp_text` e `clarity`. No treino, são aceitas somente as três classes da tarefa; linhas sem texto/rótulo e textos vazios são descartados. No teste de entrega, `clarity` precisa estar vazia. Textos vazios do teste são preservados, assim como a ordem e as demais colunas.

Os dados não são incluídos por padrão no Git. Obtenha-os pela fonte autorizada da disciplina. A remoção de um arquivo do Git não o torna dispensável para executar os scripts ou reexecutar o notebook.

## Execução

Execute os comandos a partir da raiz do EP1.

### Baseline

```powershell
.\.venv\Scripts\python.exe -u baseline_classifier.py --input Dataset/train.xlsx --test Dataset/test1.xlsx --output baseline_results/test1_rotulado.xlsx --n-jobs 1
```

A busca contém 72 combinações, cinco dobras e o reajuste do vencedor em todo o treino. Novas execuções também salvam a tabela de resultados da busca em `.cv.csv`.

### Comparação clássica

```powershell
.\.venv\Scripts\python.exe -u classical_model_selection.py --input Dataset/train.xlsx --test Dataset/test1.xlsx --output classical_selection --n-jobs 1
```

A busca atual é extensa. O paralelismo padrão é um para reduzir disputa de memória. Cada família tem seu melhor resultado registrado, e o vencedor é reajustado no treino completo.

### BERTimbau Base

```powershell
.\.venv\Scripts\python.exe -u bertimbau_classifier.py --input Dataset/train.xlsx --test Dataset/test1.xlsx --output modelo_bertimbau --batch-size 4 --gradient-accumulation-steps 4 --dataloader-workers 0
```

Os valores de lote acima são um ponto de partida conservador; ajuste conforme a memória disponível. A busca é sequencial e avalia 12 combinações em cinco dobras, além do reajuste final. Pode levar muitas horas e exige download do checkpoint pré-treinado no primeiro uso.

Os comandos gravam saídas nas pastas indicadas. Use outra pasta ou outro nome de saída para preservar resultados de execuções anteriores. Não há retomada de grid após interrupção.

## Notebook de resultados

Abra `analise_resultados.ipynb` no VS Code/Jupyter, selecione o kernel do ambiente com as dependências instaladas e execute a partir da pasta EP1.

O notebook já contém saídas da análise realizada durante a revisão e não treina os modelos. Inclui:

- Auditoria da base e distribuição das classes.
- Grupos com avaliações conflitantes.
- Ranking e gráfico das métricas históricas de seleção.
- Hiperparâmetros registrados.
- Alinhamento e validade dos rótulos da entrega existente.
- Sobreposição textual entre treino e teste.
- Limitações da avaliação.

Sem os dados locais, ainda é possível consultar as saídas salvas, mas não reexecutar todas as células.

## Arquivos grandes e Git

O `.gitignore` exclui ambientes virtuais, caches Python, modelos binários, pesos do BERTimbau, planilhas de dados e entregas geradas. Código, resumos pequenos e notebook podem ser versionados.

Preserve modelos treinados em backup externo caso queira usá-los sem repetir o treinamento. Ignorar arquivos não remove versões já presentes no histórico Git.

A revisão identificou um histórico Git associado a uma estrutura antiga com subpasta `EP1/`, enquanto os arquivos atuais estão na raiz. Nenhum commit, reset ou alteração de histórico foi realizado. Confira `git status` antes de preparar a publicação.

## Limitações e interpretação

- A CV de seleção pode produzir estimativas otimistas; uma avaliação aninhada ou um teste independente seria necessário para uma estimativa mais rigorosa.
- O agrupamento não identifica todas as paráfrases ou respostas quase idênticas.
- Duplicatas são agrupadas, mas não eliminadas automaticamente; podem afetar a ponderação das observações.
- Rótulos conflitantes e contexto ausente limitam a predição a partir do texto.
- Sobreposição entre treino e teste limita a interpretação como generalização para textos inéditos.
- Os resultados históricos não foram recriados por um novo treinamento durante a revisão.
