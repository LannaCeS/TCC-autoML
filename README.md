# Pacote para reproduzir o benchmark AutoML

Este diretório contém os insumos necessários para executar do zero o benchmark
oficial: configuração, testes e
as cinco imagens Docker exatas usadas na execução concluída. A pasta `v2/outputs/`
começa vazia; os resultados novos serão gravados ali.

A ideia é gerar uma nova execução com 200 combinações de ferramenta, cenário e seed.

## Requisitos do computador

- Linux x86-64, Docker Engine com Docker Compose v2 e Python 3 no host.
- Aproximadamente 75 GB de RAM disponível: quatro containers podem usar até
  64 GiB, e o orquestrador reserva mais 8 GiB para o host.
- Topologia de CPU compatível com os slots registrados em `v2/config.json`:
  `0-7`, `8-15`, `32-39` e `40-47`, sem sobreposição de núcleos físicos.
- Espaço livre para descompactar as imagens, executar os modelos e armazenar os
  artefatos (o benchmark completo gera muitos gigabytes).

O protocolo valida a topologia antes de começar. Uma máquina com CPUs ou
topologia diferentes pode exigir nova revisão da configuração e não reproduzirá
as condições de custo computacional do experimento original.

## 1. Carregar as imagens Docker congeladas

Na raiz deste pacote:

```bash
gzip -t docker/automl-images.tar.gz
docker load -i docker/automl-images.tar.gz
```

Confira as cinco tags e IDs:

```bash
docker image ls --format '{{.Repository}}:{{.Tag}} {{.ID}}' | grep '^lanna-automl-'
```

Os IDs esperados estão em `MANIFEST.json`. O Docker deve preservar os IDs ao
carregar o arquivo; o runner oficial recusa imagens diferentes das revisadas.

## 2. Conferir o dataset e o código

```bash
sha256sum -c SHA256SUMS
cd v2
mkdir -p outputs
python3 orchestrate.py --help
```

O hash esperado do dataset `data/creditcard.csv` é
`76274b691b16a6c49d3f159c883398e03ccd6d1ee12d9d8ee38f4b4b98551a89`.
O CSV contém 284.807 linhas e deve permanecer inalterado.

## 3. Validar o ambiente antes da execução longa

Com as imagens carregadas e a partir de `v2/`:

```bash
docker compose config --quiet
docker compose run --rm --no-deps prepare -m unittest discover -s /code/tests -v
docker compose run --rm --no-deps prepare /code/tests/validate_data.py
python3 orchestrate.py --smoke
```

O smoke test usa amostra e orçamento curto. Ele serve apenas para testar
integração, não entra nos resultados científicos. Seus arquivos ficam numa
pasta `outputs/smoke-*` separada.

## 4. Executar os 200 experimentos oficiais do zero

```bash
python3 orchestrate.py --workers 4
```

O comando oficial executa cinco ferramentas (AutoGluon, TPOT, PyCaret, FLAML e
H2O), oito cenários e cinco seeds (`42, 101, 223, 302, 404`). A métrica de
seleção é F1 macro na validação; cada busca tem teto de 1.800 segundos. O teste
externo só é usado depois de congelar o modelo selecionado. O pré-processamento
tem teto conjunto de 1.800 segundos e o pós-processamento, de 900 segundos por
caso. A execução completa pode levar mais de um dia.

Não use `--smoke`, `--qualify`, `--frameworks` ou `--scenarios` para a execução
que pretende comparar com os resultados completos do artigo. Não execute dois
orquestradores ao mesmo tempo. Para reduzir a concorrência, `--workers 1` é
aceito pelo runner, mas altera as condições de tempo e recursos do protocolo.

## 5. Acompanhar e localizar os resultados

Cada chamada cria uma nova pasta `v2/outputs/official-*`. O arquivo
`finished.json` informa se todos os casos chegaram a um estado terminal e
quantos foram concluídos. Logs, seleções, previsões, métricas, modelos e
metadados ficam em `runs/<ferramenta>-<cenário>-<seed>/`; as divisões e matrizes
preparadas ficam em `prepared/`. O snapshot congelado de código, hash do CSV e
IDs das imagens ficam junto da execução.

Para interromper com segurança, use Ctrl+C no orquestrador. Uma nova execução
começa em diretório próprio; o runner não sobrescreve resultados anteriores.
Uma interrupção não transforma casos parciais em resultados válidos.

## O que significa reproduzir

As imagens congelam as versões das bibliotecas e o pacote preserva código,
configuração, dados e seeds. Ainda assim, algumas bibliotecas, em especial o
H2O/Deep Learning, podem ter não determinismo numérico. Diferenças de CPU,
topologia, carga do sistema e cache também afetam tempos e, ocasionalmente,
ordenações em empates. Portanto, uma nova execução reproduz o protocolo e a
configuração; não há garantia de hashes de modelos ou métricas bit a bit idênticos.

Para a descrição completa das partições, samplers, seleção, recursos e
limitações, consulte [`v2/README.md`](v2/README.md).
