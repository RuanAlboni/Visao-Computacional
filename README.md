# Smart Fridge Vision — estado da aplicação e guia de execução

Grupo 7 — Albano, Ruan e Rafael  
Estado consolidado em: 6 de novembro de 2026.

Este documento descreve a versão multialimento e os resultados apresentados até esta data. Não representa uma auditoria da branch atual do repositório: arquivos instalados, versões das bibliotecas e integração final devem ser conferidos no ambiente local.

## 1. Documentação do projeto

[Consultar a documentação completa no Google Docs](https://docs.google.com/document/d/1XyEe_k_wufmqPo96Uion9d9OsMO1fAW_dub4dYEo40A/edit?usp=sharing)

O endereço foi fornecido pela equipe. Seu conteúdo online não pôde ser verificado nesta consulta; a síntese abaixo utiliza a documentação e os testes compartilhados na conversa.

## 2. Estado atual da aplicação

Legenda:
- Implementado/demonstrado: recurso presente no fluxo com evidência de execução apresentada.
- Implementado: recurso disponibilizado em código, sem avaliação completa de uso ou desempenho.
- Preparado: estrutura disponível, mas integração ou execução real ainda pendente.
- Pendente: capacidade ainda não demonstrada ou validada.

| Componente ou funcionalidade | Estado | O que existe atualmente | Pendência ou observação |
|---|---|---|---|
| Painel multialimento em Streamlit | Implementado/demonstrado | Aplicação `app_alimentos.py` para envio e análise de imagens. | Conferir a versão dos arquivos no ambiente de cada integrante. |
| Leitura e preparação da imagem | Implementado | Leitura, orientação EXIF quando disponível, conversão RGB e limite de tamanho. | Não equivale a tratamento de todas as condições de captura. |
| Detector compartilhado YOLO11n | Implementado/demonstrado | Carregamento local de `models/yolo11n.pt` e produção de caixas, categorias e scores. | Não houve treinamento ou ajuste fino próprio demonstrado. |
| Catálogo de alimentos | Implementado/demonstrado | Cenoura, brócolis, banana, maçã e laranja. | A disponibilidade das classes não garante reconhecimento em qualquer cena. |
| Filtro por categoria | Implementado/demonstrado | Seleção de um subconjunto sem carregar uma rede por alimento. | O filtro não treina nem melhora o modelo. |
| Tratamento das caixas | Implementado | Verificação das classes, limites da imagem, área válida e atribuição de IDs. | Não corrige automaticamente falsos positivos ou agrupamentos. |
| Cores e identificação | Implementado/demonstrado | Cor fixa por produto e IDs associados às caixas e aos resultados. | Cor representa categoria, não condição de conservação. |
| Resumo por categoria | Implementado/demonstrado | Quantidade de caixas e IDs por alimento. | Não é uma contagem validada de unidades. |
| Galeria de recortes | Implementado | Conferência de cada região com categoria, score e estado. | Avaliar se cada recorte contém um objeto ou um grupo. |
| Exportação PNG e JSON | Implementado | Imagem anotada e relatório vinculados ao ID da análise. | Preservar os arquivos junto das imagens e configurações dos testes. |
| Logs e rastreabilidade | Implementado; metadados demonstrados | ID, hash da imagem, parâmetros e identificação/hash dos pesos; log rotativo. | Acrescentar versões, hardware e versão do código ao protocolo de avaliação. |
| Download do detector | Implementado | Script explícito para obter o checkpoint oficial e verificar as classes. | Requer internet na primeira obtenção; pesos não acompanham o código. |
| Adaptador `CarrotModule` | Implementado | Adaptação ao contrato comum usando o motor compartilhado. | A integração completa com o orquestrador geral não foi demonstrada. |
| Classificador multiclasse | Preparado | Carregador para checkpoint compatível, inferência sobre recortes e tratamento de resultados. | Falta obter, conferir e integrar o checkpoint real de Ruan. |
| Estado de banana, maçã e laranja | Pendente de integração | Ruan informou possuir um modelo para essas categorias. | As análises apresentadas registram apenas o detector e `nao_avaliado`. |
| Estado de cenoura e brócolis | Pendente | O fluxo evita inventar uma condição sem classes compatíveis. | Exige pesos e dados adequados para essas categorias. |
| Treinamento multiclasse | Preparado | Script com pastas por classe, ResNet-18, validação e seleção por F1 macro. | Não foi apresentada execução real desse treinamento. |
| Aba de treinamento | Implementado | Gera um comando para execução no terminal. | Não inicia treinamento automaticamente. |
| Abordagem clássica para as cinco categorias | Pendente | Não é o caminho principal da aplicação demonstrada. | Ainda não há método clássico completo e avaliado para esse escopo. |
| Testes de software | Verificação parcial registrada | Dez testes com backend simulado foram aprovados durante a preparação do pacote. | Executar a suíte no ambiente local; não mede acurácia das redes. |
| Avaliação quantitativa | Pendente | Há inspeção funcional e visual de exemplos. | Construir conjunto anotado e medir erros de detecção, classificação e contagem. |
| Prazo restante de conservação | Não validado | Campo permanece sem estimativa no fluxo apresentado. | Não converter score diretamente em dias. |
| Segurança alimentar | Fora do alcance validado | Avisos distinguem aparência visual de confirmação de segurança. | O protótipo não determina se um alimento é seguro para consumo. |
| Vídeo, sensores e inventário persistente | Não demonstrado | Operação atual por imagens enviadas pelo usuário. | Integrações futuras, caso entrem no escopo. |

### 2.1 Resultados funcionais registrados

| Imagem | Categorias selecionadas | Caixas previstas | Tempo registrado, sem carregar pesos |
|---|---|---|---:|
| `cenoura_3.jpg` | Cenoura | Nenhuma caixa | 33,1448 s |
| `frutas-e-vegetais.webp` | Cinco categorias | 2 de banana, 1 de maçã e 3 de laranja | 0,0725 s |
| `frutas-legumes-da-estacao.jpg` | Cinco categorias | 1 de cenoura e 5 de brócolis | 0,1174 s |

Configuração registrada: `conf = 0.35`. Não houve classificação de condição nesses testes.

Interpretação:
- Uma caixa de cenoura abrangeu um agrupamento, não uma unidade isolada.
- As duas caixas de banana apresentaram sobreposição importante.
- Uma previsão de laranja foi considerada visualmente suspeita.
- Zero detecções não comprova ausência do alimento.
- Esses tempos não são um benchmark e não demonstram operação em tempo real. Não atribuir diferenças a cache, hardware ou aquecimento sem medição controlada.

## 3. O que Ruan precisa informar e disponibilizar

Ruan precisa fornecer o checkpoint e os metadados necessários para reproduzir a inferência e documentar o treinamento. Não basta informar que o modelo reconhece três frutas: é necessário conhecer exatamente como suas saídas devem ser interpretadas.

| Informação | O que deve ser fornecido | Por que é necessário |
|---|---|---|
| Checkpoint | Arquivo dos pesos e caminho ou local de obtenção. | Carregar o modelo real, não apenas uma arquitetura vazia. |
| Código e versão | Branch/commit e código de definição e carregamento da rede. | Reproduzir a arquitetura e evitar incompatibilidade de camadas. |
| Arquitetura | Confirmar ResNet-18, CNN ou outra, incluindo alterações na saída. | Construir o modelo correspondente ao `state_dict`. |
| Formato do arquivo | Informar se contém `state_dict`, dicionário de checkpoint ou outro formato. | Adequar o carregador sem adivinhar a estrutura. |
| Classes e índices | Lista completa e ordenada das classes. | Evitar associar um índice à fruta ou condição errada. |
| Pré-processamento | Tamanho, RGB/BGR, recorte, redimensionamento, média e desvio da normalização. | Reproduzir as mesmas condições de entrada do treinamento. |
| Origem dos dados | Nome/link da base, condições de uso e forma de coleta. | Documentar procedência e possibilidade de reprodução. |
| Distribuição | Quantidade de imagens por classe. | Identificar cobertura e desequilíbrios. |
| Rótulos de condição | Critério usado para fresco/deteriorado e exemplos. | Definir o significado real da saída. |
| Divisão dos dados | Quantidades e critérios de treino, validação e teste. | Verificar independência e risco de vazamento entre conjuntos. |
| Inicialização | ImageNet, outro pré-treinamento ou início aleatório. | Documentar corretamente transferência de aprendizado. |
| Configuração de treino | Épocas, lote, otimizador, taxa de aprendizado, aumentos e seleção do checkpoint. | Reproduzir e explicar o treinamento. |
| Avaliação | Métricas por classe, matriz de confusão e conjunto utilizado. | Distinguir desempenho medido de exemplos pontuais. |
| Ambiente | Versões de Python, PyTorch/Torchvision e dispositivo. | Reproduzir carregamento e inferência. |
| Exemplos de controle | Imagens de entrada e saídas esperadas. | Conferir o adaptador antes de conectar os recortes do detector. |

### 3.1 Contrato esperado pelo carregador preparado

O carregador da entrega atual espera um dicionário com:

```text
arquitetura   # por exemplo: resnet18
classes       # lista ordenada das classes
modelo        # state_dict compatível com a arquitetura
tamanho       # tamanho de entrada
```

Os valores reais devem ser conferidos no arquivo.

Se o checkpoint tiver outro formato, adaptar o carregador com base no código de Ruan. Não renomear o arquivo ou reordenar classes para tentar eliminar erros.

O pré-processamento padrão do adaptador foi preparado para redimensionamento quadrado, entrada RGB e normalização ImageNet. Isso não comprova que o treinamento de Ruan utilize o mesmo procedimento.

## 4. Guia de execução

### 4.1 Pré-requisitos

- Projeto multialimento e ajuste visual já aplicados.
- Python e ambiente virtual compatíveis com as dependências.
- Terminal aberto na raiz do projeto.
- Internet para instalar pacotes e obter os pesos, quando necessário.
- Checkpoints de origem confiável.

Arquivos esperados da entrega:

```text
app_alimentos.py
requirements-alimentos-v2.txt
src/alimentos_v2/catalogo.py
src/alimentos_v2/fluxo.py
src/alimentos_v2/profundo.py
src/alimentos_v2/visual.py
scripts/baixar_detector_alimentos.py
scripts/inspecionar_checkpoint_alimentos.py
scripts/treinar_multiclasse.py
tests/test_alimentos_v2.py
```

Caso algum arquivo esteja ausente, aplicar a entrega correspondente antes de continuar. Este guia não pressupõe que os arquivos estejam presentes em todas as branches.

### 4.2 Acessar o projeto e ativar o ambiente

No ambiente já utilizado por Albano:

```bash
cd ~/smart_fridge_vision
source venv/bin/activate
```

Em outro computador, ajustar o caminho e o nome do ambiente virtual.

Somente se ainda não existir ambiente virtual:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4.3 Instalar as dependências

```bash
python -m pip install -r requirements-alimentos-v2.txt
```

Se ocorrer incompatibilidade no Python 3.14, registrar a mensagem completa e conferir suporte das versões de PyTorch, Torchvision e Ultralytics. Não remover ou trocar bibliotecas do ambiente indiscriminadamente.

### 4.4 Obter o detector

Se `models/yolo11n.pt` ainda não estiver disponível:

```bash
python scripts/baixar_detector_alimentos.py
```

O script realiza um download explícito, verifica as categorias relevantes e disponibiliza o checkpoint em:

```text
models/yolo11n.pt
```

Se o detector já estiver presente e funcionando, não é necessário baixá-lo novamente. Verificar as condições de licença aplicáveis ao uso do software e dos pesos.

### 4.5 Executar os testes

Teste específico da entrega multialimento:

```bash
python -m pytest -q tests/test_alimentos_v2.py
```

Para conferir também os demais testes existentes no projeto:

```bash
python -m pytest -q tests/
```

A suíte geral pode incluir módulos de outras versões. Registrar separadamente suas falhas e as do fluxo multialimento. Testes com respostas simuladas não demonstram qualidade de detecção em fotografias.

### 4.6 Iniciar o painel

```bash
python -m streamlit run app_alimentos.py
```

Abrir o endereço local informado pelo Streamlit. Utilizar `Ctrl+C` no terminal para encerrar.

O ponto de entrada desta versão é `app_alimentos.py`. O `app.py` anterior não representa necessariamente o mesmo fluxo.

### 4.7 Primeira análise: somente detecção

1. Enviar uma imagem.
2. Selecionar todas as categorias ou um subconjunto.
3. Informar `models/yolo11n.pt` como detector.
4. Manter o classificador de condição desabilitado.
5. Usar inicialmente `conf = 0.35`, preservando a configuração dos exemplos.
6. Selecionar CPU, ou CUDA apenas se o ambiente tiver suporte configurado.
7. Clicar em `Analisar`.
8. Conferir cores, caixas, IDs, resumo e recortes.
9. Exportar PNG e JSON para registrar o caso.

| Cor | Categoria |
|---|---|
| Ciano | Cenoura |
| Verde | Brócolis |
| Amarelo | Banana |
| Rosa | Maçã |
| Laranja | Laranja |

As cores indicam alimento, não conservação. Na ausência de classificador, `nao_avaliado` é o comportamento esperado.

### 4.8 Inspecionar o checkpoint de Ruan

Após receber o arquivo e confirmar que é de origem confiável:

```bash
python scripts/inspecionar_checkpoint_alimentos.py \
  "/caminho/para/checkpoint_de_ruan.pth"
```

Conferir as chaves, a arquitetura, as classes e o tamanho. Se o formato não corresponder ao contrato, adaptar o carregador antes de habilitar a classificação.

### 4.9 Executar detecção e classificação

1. Confirmar o carregamento do checkpoint em um exemplo de controle.
2. Habilitar `Usar classificador multiclasse de estado` no painel.
3. Informar o caminho do arquivo.
4. Clicar novamente em `Analisar`.
5. Conferir se o bloco `modelos` do JSON registra o classificador e suas classes.
6. Conferir estado, classe e score em cada recorte aplicável.

Com um checkpoint limitado a banana, maçã e laranja, cenoura e brócolis devem continuar sem avaliação de condição.

O limiar de estado padrão preparado é `0.60`. Sua presença no formulário ou JSON não comprova que a classificação executou. Um resultado visual fresco não confirma segurança para consumo.

### 4.10 Treinamento multiclasse — opcional e ainda não validado

Executar somente após obter dados rotulados, definir as classes e separar os conjuntos:

```text
data/alimentos/
├── train/
│   ├── banana_fresca/
│   ├── banana_deteriorada/
│   ├── maca_fresca/
│   ├── maca_deteriorada/
│   ├── laranja_fresca/
│   └── laranja_deteriorada/
├── val/
│   └── mesmas classes, com imagens independentes
└── test/
    └── mesmas classes, reservadas para avaliação final
```

Não distribuir fotos quase iguais do mesmo alimento físico entre treino, validação e teste.

```bash
python scripts/treinar_multiclasse.py \
  --dados data/alimentos \
  --saida models/alimentos_multiclasse.pth \
  --epocas 15 \
  --device cpu
```

O script preparado usa pré-treinamento por padrão, podendo baixar pesos ImageNet. A opção `--sem-pretreino` evita esse pré-treinamento, mas altera o procedimento de treinamento.

A seleção do checkpoint utiliza a validação; o conjunto de teste não é avaliado automaticamente por esse script. A avaliação final precisa ser implementada e executada separadamente. A aba do painel apenas gera o comando.

### 4.11 Consultar logs e resultados

Log do fluxo compartilhado:

```text
logs/alimentos_v2.log
```

PNG e JSON são disponibilizados pelos botões de exportação do painel. Conferir o mesmo ID em todos os arquivos e preservar os registros junto da imagem utilizada.

## 5. Diagnóstico rápido

| Situação | O que conferir |
|---|---|
| Detector não encontrado | Caminho local e execução do script de download. |
| Zero detecções | Registro de execução, categorias selecionadas, imagem original e previsões; não interpretar automaticamente como ausência. |
| Estado `nao_avaliado` | Se o classificador está habilitado, carregado e cobre o alimento. |
| Classes ou camadas incompatíveis | Formato e arquitetura real do checkpoint; solicitar o código a Ruan. |
| Resultado `inconclusivo` | Discordância de produto ou score abaixo do limiar, conforme o motivo registrado. |
| Quantidade diferente da imagem | Agrupamentos, sobreposições, falsos positivos e objetos não detectados. |
| Execução lenta | Repetir medições controladas; registrar dispositivo, versões, resolução e tempos das etapas. |
| Falha no ajuste visual | Conferir se `visual.py` e `app_alimentos.py` correspondem à mesma entrega. |

## 6. Próxima entrega prioritária

Integrar e verificar o checkpoint de Ruan antes de ampliar o escopo. Em seguida, avaliar detector e classificador com dados independentes, mantendo separadas contagem de caixas, unidades reais e condição visual.
