# Benchmark AutoML — versão 2

Esta versão (protocolo 2.4) implementa o protocolo: comparar as ferramentas AutoML,
selecionar os modelos por **F1 macro**, reservar até **1.800 segundos de busca** por
ferramenta/cenário/seed e usar as seeds `42, 101, 223, 302, 404`.

## Partições e seleção

1. A preparação valida rótulos, valores ausentes, valores não finitos e o hash
   SHA-256 do CSV. Os grupos são definidos pelas linhas de atributos idênticas,
   sem incluir o rótulo. Uma verificação detecta eventuais colisões do hash.
2. Dois sorteios estratificados dos grupos únicos definem teste externo de 20%
   dos grupos e validação interna de 20% dos grupos restantes. Os grupos são
   depois expandidos para suas linhas originais: aproximadamente 64% treino,
   16% validação e 20% teste. A estratificação é feita por classe no nível dos
   grupos; percentuais por linha e prevalências realizados são registrados,
   pois grupos têm tamanhos diferentes. Grupos com rótulos conflitantes provocam
   erro explícito, sem relabeling ou separação indevida das cópias.
3. Há **uma validação interna comum**, não uma CV de cinco folds. As cinco seeds
   repetem esse protocolo. Todos os candidatos usam treino e validação com grupos
   separados; o teste externo fica ausente do arquivo enviado à busca.
4. `StandardScaler` é ajustado somente no treino. Balanceamento e pesos também
   são calculados somente no treino. As quatro ferramentas fora do H2O
   reutilizam as matrizes comuns. O H2O recebe a variante descrita abaixo,
   necessária para separar o treino dos modelos-base e o blending sem vazamento.
   Índices, prevalências, versões e checksums são preservados em ambas.
5. A ferramenta seleciona por `(F1_classe_0 + F1_classe_1) / 2`, sem ponderação
   pelo tamanho das classes. A seleção e sua avaliação na validação são gravadas
   antes da primeira previsão no teste externo.
6. Todas as ferramentas recebem o mesmo conjunto original de treino, validação
   e teste. O H2O divide internamente o treino entre modelos-base e blending.
   TPOT e PyCaret fazem um refit explícito da instância
   selecionada nessas linhas, pois seus modelos finais internos usam também a
   validação. Esse refit explícito é cronometrado separadamente. O tempo de busca
   inclui os refits internos descartados, limitação registrada nos metadados.

Os thresholds não são ajustados no teste. Usa-se `predict` nativo, com o padrão
0,5 nos classificadores probabilísticos, sem calibração automática de threshold
no AutoGluon. Average Precision usa probabilidades da classe 1 ou margens de decisão com a
orientação correta e precisão completa. Se nenhum score contínuo estiver
disponível, AP é registrado como indisponível, sem substituição por rótulos.

## Ferramentas e cenários

| Ferramenta | Busca e seleção por F1 macro | Pesos por classe |
| --- | --- | --- |
| AutoGluon | `eval_metric='f1_macro'`, HPO aleatório local, validação explícita, sem bagging/stacking com partições privadas | Pesos por amostra; avaliação não ponderada |
| FLAML | Callback com perda `1 - F1_macro`; validação explícita | Pesos por amostra; avaliação não ponderada |
| PyCaret | Métrica customizada; comparação e tuning Optuna/TPE usam o mesmo `PredefinedSplit` | Pesos balanceados no classificador em cada ajuste; avaliação não ponderada |
| TPOT | Scorer explícito e `PredefinedSplit`; pipelines sem classificadores internos com CV privada | Pesos balanceados no classificador final em cada ajuste; avaliação não ponderada |
| H2O AutoML | Métrica customizada `1 - F1_macro`; modelos-base e Stacked Ensembles classificados na validação externa ao blending | Pesos nos modelos-base; blending e avaliação não ponderados |

No H2O 3.46.0.12 instalado, F1 macro é implementado por `custom_metric_func`.
A função acumula TP, TN, FP e FN antes de calcular a média dos dois F1s e retorna
a perda `1 - F1_macro`, que o AutoML minimiza na ordenação e, nos algoritmos
compatíveis, no critério de parada. Ela usa
`p1 > 0.5` (empate atribuído à classe 0), sem o threshold automático que maximiza o F1 da classe positiva.
Os scores são comparados com um cálculo independente usando scikit-learn.
O protocolo permite **GBM, DRF, XGBoost, GLM, Deep Learning e Stacked Ensembles**.
A CV privada fica desativada (`nfolds=0`), mas os ensembles são habilitados com
um `blending_frame` explícito. Na imagem instalada,
testes separados de cada família confirmaram scores customizados iguais ao
cálculo do scikit-learn. GBM, DRF, XGBoost e Deep Learning mantiveram `custom`
nos parâmetros de parada; GLM manteve `AUTO`. Assim, o critério principal de
ranking é F1 macro em todas essas famílias, mas a parada interna do GLM é uma
exceção registrada em `selection.json`. Não se afirma que cada passo interno de
treino ou de busca do GLM minimize diretamente a perda macro.

### Blending do H2O

Antes de qualquer escala ou balanceamento, é feito um sorteio estratificado por grupos
reserva **20% do treino original** para o metalearner dos ensembles. Isso
corresponde aproximadamente a 51,2% do dataset para os modelos-base, 12,8% para
blending, 16% para validação e 20% para teste. Os percentuais exatos e contagens
de fraude são registrados para cada seed. Grupos de registros idênticos não
atravessam nenhuma dessas quatro partições.

O scaler do H2O é ajustado só nos modelos-base. Os samplers e pesos balanceados
também usam apenas essa parte. Blending, validação e teste são transformados com
esse scaler; o blending não é reamostrado e recebe pesos iguais a um. Assim,
nem atributos nem rótulos de blending participam do ajuste do scaler/sampler
dos modelos-base. Essa preparação própria fica em `data_h2o.npz`; as matrizes
das outras ferramentas permanecem em `data.npz`. Os mesmos IDs e rótulos de
validação/teste são usados por todas as ferramentas, embora a escala do H2O
seja específica do seu treino-base.

`AllModels` e `BestOfFamily` passam a ser candidatos, junto dos modelos-base.
O vencedor continua sendo escolhido pelo F1 macro da validação: não se força
um ensemble a vencer. Pelo menos um ensemble deve ser criado; ausência resulta
em falha explícita, em vez de uma execução apresentada como stacking bem-sucedido.
Os ensembles e o vencedor são salvos. O orçamento total de busca continua em
1.800 segundos, incluindo a criação e persistência dos ensembles.

A diferença de arquitetura precisa integrar o fato de que o H2O usa blending e os
outros adaptadores mantêm suas configurações. Esta é uma comparação dessas
configurações dos sistemas AutoML, sem supor espaços ou arquiteturas idênticos.

A documentação geral restringe parada customizada em Python a GBM/DRF. Como a
imagem instalada demonstrou suporte adicional, seu ID e versões fazem parte do
protocolo; esse comportamento não é presumido em outra instalação. O teste por
família está em `outputs/h2o-algorithm-probe/report.json`. Deep Learning pode
apresentar não determinismo apesar da seed, o que limita reprodução bit a bit.

A integração anterior havia excluído o H2O por não encontrar F1 macro na lista
de métricas documentadas. Um teste executável da API instalada confirmou o
suporte a `custom_metric_func` junto de `sort_metric='custom'`; a exclusão foi
corrigida. O teste e seu relatório estão em `outputs/h2o-macro-probe/`.

Os cenários configurados são: baseline, RandomOverSampler, SMOTE,
EditedNearestNeighbours, TomekLinks, SMOTEENN, SMOTETomek e pesos balanceados por
classe. O cenário de pesos se chama `class_weighted`: não representa um modelo
de custos financeiros. O espaço de candidatos de cada ferramenta permanece
igual entre cenários suportados. Os espaços diferem entre ferramentas, como é
esperado em uma comparação de sistemas AutoML.
O AutoGluon recebe um gerador de identidade previamente ajustado somente no
treino: os atributos já são numéricos e escalados. Isso impede que o gerador
padrão seja ajustado sobre a concatenação interna de treino e validação.
O FLAML não aumenta classes automaticamente (`auto_augment=False`). Baseline
significa ausência de balanceamento externo; os espaços nativos ainda podem
conter opções de ponderação próprias dos algoritmos.
Na imagem instalada, os scores armazenados de RF/ExtraTrees divergiram das
previsões novas sobre a mesma validação. Por isso o adaptador reavalia todos os
candidatos ajustados nessa validação e escolhe pelo F1 macro observado; esse
custo fica dentro do teto de busca. A tabela nativa é preservada separadamente.
Após recarregar um modelo sem refit, uma nova divergência invalida a execução.
O AutoGluon usa os espaços de HPO nativos da versão instalada; RF e ExtraTrees,
que não expõem um espaço nativo nesse adaptador, treinam a configuração fornecida.

No PyCaret, até 30% do orçamento nativo (máximo de oito minutos) é destinado à
comparação inicial. O tempo restante ajusta os hiperparâmetros da melhor família
com Optuna TPE e avaliações nativas do PyCaret isoladas por processo. O tuning
só substitui o checkpoint se melhorar seu F1 macro de validação.
`setup` recebe treino+validação para possibilitar o `PredefinedSplit`, mas
`preprocess=False` impede transformações ajustadas antes do fold. O holdout
auxiliar do PyCaret repete a validação e não é usado como teste externo.

### Pesos no PyCaret e TPOT — protocolo 2.4

Todas as 200 combinações ferramenta/cenário/seed são suportadas. O PyCaret usa
LR, DT, RF, ExtraTrees, GradientBoosting, AdaBoost, GaussianNB e LightGBM.
O TPOT mantém o template linear-light, com BernoulliNB, DT, GaussianNB, LR e
MultinomialNB como classificadores finais. KNN é removido de ambas as ferramentas
**em todos os cenários**, preservando as mesmas famílias com e sem pesos.

`BalancedClassifier` calcula `n_treino / (2 * n_classe_treino)` a partir do alvo
recebido em cada `fit` e encaminha esse vetor ao `sample_weight` do classificador.
Apenas o cenário `class_weighted` ativa esses pesos; os demais usam a mesma
camada sem pesos. A transformação de atributos continua ajustada no treino e
não recebe pesos extras. Não há reamostragem usada para simular ponderação.
Classificadores sem suporte explícito a pesos provocam erro.

O PyCaret recebe oito classificadores customizados com essa camada. Seus espaços
nativos de tuning são preservados com o prefixo `estimator__`, excluindo opções
internas de balanceamento por classe. Isso evita aplicar pesos por amostra e
`class_weight` simultaneamente. O TPOT envolve todos os classificadores finais
na mesma camada e neutraliza a constante `class_weight='balanced'` da configuração
nativa de LogisticRegression, em todos os cenários. Não há classificadores internos
ou CV privada. Pesos são calculados após a separação do `PredefinedSplit`.

A versão instalada do PyCaret apresentou erro no encaminhamento direto de
`fit_kwargs={'sample_weight': ...}`; a camada dentro do classificador permite
aplicar pesos durante comparação, tuning e refit sem depender dessa rota.
Na implementação anterior, registrar a camada no catálogo interno era necessário
para o `tune_model` encaminhar o timeout ao Optuna. Ainda assim, esse timeout
não interrompia avaliações longas e ajustes finais. Na revisão 2.4.3, o adaptador
agenda avaliações nativas do PyCaret com prazos por processo e utiliza as
distribuições nativas no Optuna TPE. Refits em treino+validação são desativados.
O modelo final é
reajustado apenas no treino original e salva `training-weight-audit.json`, com
contagens de classes, quantidade de linhas e massa total dos pesos por classe.
O runner exige essa auditoria antes de avaliar o modelo final. A avaliação e a
seleção continuam usando F1 macro **sem pesos**.

O PyCaret conserva o melhor candidato concluído antes dos prazos, sem esperar
uma tentativa longa terminar. O teto externo continua em 1.800 segundos e a
etapa nativa usa 90% desse tempo. Ausência de candidato concluído é falha
registrada, sem avaliação do teste externo.

A preparação dos samplers ocorre na imagem PyCaret. As quatro ferramentas fora
do H2O recebem as mesmas linhas sintéticas/removidas. No H2O, a mesma técnica é
ajustada ao subconjunto dos modelos-base, preservando o blending. Os tempos de
cada variante são informados separadamente, sem incluir o preparo da outra;
o custo compartilhado de leitura e divisão é identificado nos manifests.
O custo de preparação não fica oculto no tempo de busca. A etapa conjunta tem limite de
1.800 segundos; exceder esse limite é falha explícita do cenário, não resultado
de desempenho. Não há garantia de que todos os samplers terminarão nesse prazo.

## Docker e execução

As imagens `lanna-automl-<ferramenta>:local` foram preparadas nesta máquina na
etapa anterior. O Compose reutiliza essas imagens, sem instalar pacotes no host.
Para reproduzir em outra máquina, também é necessário disponibilizar essas
imagens; os manifests das execuções identificam seus IDs e pacotes.

Execute dentro de `v2/`:

```bash
mkdir -p outputs
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose config --quiet
docker compose run --rm --no-deps prepare -m unittest discover -s /code/tests -v
python3 orchestrate.py --smoke
```

Cada chamada congela fontes/configuração em `source/`, copia o CSV para
`input/` e fixa as imagens pelo ID, sem depender de alterações posteriores dos
arquivos ou das tags. Checksums são conferidos antes das etapas; dados preparados
com outra receita são rejeitados. As execuções oficiais exigem `REVIEW.json`
com os hashes do código revisado. Alterar o código ou a configuração invalida
essa revisão e exige nova verificação antes de executar oficialmente. O CSV e
os IDs das imagens também devem coincidir com o ambiente validado.

O teste `--smoke` usa uma amostra por grupos, uma seed e 120 segundos de busca;
a prevalência da amostra é diferente da oficial. Seus resultados ficam em
`outputs/smoke-*/` e **não devem entrar na análise científica**.

Para verificar as cinco ferramentas com o volume completo, usando apenas a
validação e sem gerar previsões ou métricas do teste externo:

```bash
python3 orchestrate.py --qualify
```

Esse modo usa até 60 segundos de busca nativa e até 300 segundos para o processo
de busca, permitindo inicializações e refits internos. É um teste técnico, sem
valor para comparar desempenho. Seus registros usam status `qualified`.

Para os experimentos oficiais, com o dataset completo, cinco seeds e 30 minutos:

```bash
python3 orchestrate.py
```

Para selecionar um subconjunto previamente declarado:

```bash
python3 orchestrate.py --frameworks flaml autogluon --scenarios baseline smote
python3 orchestrate.py --prepare-only --scenarios baseline --seed 42
```

Para conferir a independência das quatro partições e os samplers:

```bash
docker compose run --rm --no-deps prepare /code/tests/validate_data.py
```

O teste `tests/verify_h2o_stacking.py --run /outputs/<execução>/runs/<experimento>`
recarrega cada ensemble em um armazenamento de modelos H2O vazio e verifica
suas previsões e a quantidade de linhas de treino do metalearner. Ele usa
somente a validação e não altera a seleção oficial do vencedor.

O padrão é executar **quatro containers simultâneos**, incluindo preparação.
Cada vaga recebe oito CPUs lógicas e 16 GiB: `0-7`, `8-15`, `32-39`, `40-47`.
Nesta máquina isso corresponde a quatro núcleos físicos com SMT por vaga,
sem compartilhar núcleos físicos entre vagas. As vagas distribuem-se entre os
dois sockets. A validação de inicialização rejeita CPUs sobrepostas, irmãos
SMT em vagas distintas, CPUs indisponíveis ou memória insuficiente. Reserva-se
pelo menos 8 GiB além da soma dos limites dos containers. O consumo máximo
configurado dos quatro containers é 64 GiB.

Cada combinação cenário/seed é preparada uma única vez. As ferramentas só
entram na fila depois da preparação terminar e reutilizam os mesmos arquivos.
Preparações e experimentos compartilham o limite de quatro vagas. Quando uma
execução termina, outra assume a vaga; arquivos e logs são distintos por job.
Os IDs das CPUs, a vaga e a afinidade observada ficam registrados. Um lock
impede dois orquestradores simultâneos; use `--workers`, em vez de abrir
orquestradores independentes.

```bash
python3 orchestrate.py --workers 4
# Execução sequencial, quando necessária:
python3 orchestrate.py --workers 1
```

A execução paralela compartilha memória, cache, disco e condições térmicas.
Afinidade e limites por container reduzem a interferência, mas não a eliminam.
Resultados de tempo e quantidade de candidatos devem ser descritos como obtidos
nessa configuração concorrente, sem misturá-los aos testes sequenciais anteriores.
Os tempos não representam benchmarks de latência de uma ferramenta isolada.
Outras cargas da máquina ainda podem interferir. O protocolo completo preserva
as condições e o número de workers para interpretar essa limitação.
O orçamento é um teto,
não uma exigência de consumir exatamente 30 minutos. A busca nativa recebe 90%
desse teto para reservar tempo a inicialização/finalização; um subprocesso aplica
o limite total de 1.800 segundos, incluindo imports, busca, refits nativos e
persistência inicial. Exceder o teto resulta em `budget_exceeded`, sem score de
teste. O TPOT usa 75% do orçamento nativo para otimização e limita cada avaliação
a até 180 segundos, reservando margem para seu refit interno. O H2O também
reserva uma margem para persistência dos ensembles.
O refit explícito e a inferência são cronometrados separadamente, em outro
processo com limite de 900 segundos. Estouro produz `postprocessing_timeout`
e exclui o caso da comparação, mesmo que haja arquivos parciais. O container
tem ainda um limite externo de busca + pós-processamento + 120 segundos.
Também são salvos o tempo de carregamento do modelo, o tempo total do experimento
e o tempo observado pelo host para iniciar, executar e encerrar seu container.

Não reinicie um job no mesmo diretório: o runner recusa sobrescritas. Cada chamada
do orquestrador gera uma pasta nova. Ctrl+C ou SIGTERM encerram todos os containers ativos deste orquestrador; buscas
interrompidas não são retomadas automaticamente. Para uma parada externa, use
o filtro do projeto v2, sem afetar outros containers da máquina:

```bash
docker ps -q --filter label=com.docker.compose.project=lanna-automl-v2 | xargs -r docker stop -t 10
```

## Resultados e limites das conclusões

Cada execução guarda o protocolo, hashes das fontes, ID das imagens e manifests
de preparação. Cada experimento guarda status, logs, versões dos pacotes, modelo
selecionado, leaderboard de validação, seleção congelada, métricas de ambas as
classes, matriz de confusão e previsões por registro do teste. Um baseline que
sempre prevê a classe majoritária também é calculado.

`resources.json` lê `memory.peak`, `memory.current`, `memory.max` e `cpu.stat` do
cgroup do container, incluindo processos filhos. Esses valores cobrem a execução
inteira e não são apresentados como memória exclusiva do treino. Campos não
disponíveis no ambiente ficam nulos.

`summary.csv` inclui falhas, incompatibilidades e combinações ainda não iniciadas,
conforme o plano de execução. Métricas parciais de casos que falharam são excluídas.
`comparison-coverage.json` informa as seeds concluídas por ferramenta e a
interseção disponível para comparações pareadas; subconjuntos com falhas não
devem ser apresentados como comparação completa. Somente casos `completed`
no modo `official` são elegíveis para a análise oficial. `descriptive-summary.json`
apresenta média e desvio-padrão descritivos, com a quantidade de seeds concluídas.
As cinco repetições compartilham observações; não são tratadas como amostras
independentes para calcular p-valores ou intervalos por `desvio/sqrt(5)`.

Este é um estudo comparativo em um dataset, sob uma hipótese de divisão aleatória
por grupos. Ele não comprova generalização temporal, superioridade universal ou
benefício financeiro. Falhas, timeouts e cenários não suportados devem integrar
o relato do estudo, inclusive quando impedirem uma comparação completa.

Referências de implementação: [scikit-learn — divisão estratificada](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html),
[PyCaret — classificação](https://pycaret.readthedocs.io/en/latest/api/classification.html),
[H2O — API AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-py/docs/_modules/h2o/automl/_estimator.html),
[H2O — métricas customizadas de parada](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/algo-params/stopping_metric.html).
