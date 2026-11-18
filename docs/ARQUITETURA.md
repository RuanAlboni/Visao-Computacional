# Arquitetura do Smart Fridge Vision

## Pipeline ativo

1. `app_alimentos.py` le a imagem, EXIF, configurações e chama `Motor.analisar`.
2. `src/alimentos_v2/fluxo.py` carrega **YOLO11n** e pede bounding boxes para classes COCO: `carrot`, `broccoli`, `banana`, `apple`, `orange`. O filtro de classes vem do catalogo. `agnostic_nms=True` esta habilitado por padrao.
3. Para cada regiao detectada, o fluxo recorta a imagem RGB.
4. **Clássico**: `src/alimentos_v2/classico.py` chama `classico_core.py`, faz segmentação e pontuacao usando cor, manchas e textura. A especie vem do YOLO, nao da ResNet. A regra utiliza um limiar por fruta (`models/limiares_classico.json`). A pontuacao **nao e probabilidade**.
5. **IA**: `src/alimentos_v2/profundo.py` carrega o checkpoint real `models/alimentos_multiclasse.pth` (ResNet18). O processamento usa 224x224 RGB, normalizacao ImageNet e softmax sobre 6 classes. Se especie prevista e especie do detector discordarem, ou se o score for inferior ao limiar escolhido, o estado e `inconclusivo`. A pontuacao softmax **nao e confianca calibrada**.
6. Cenoura e brocolis: apenas deteccao; condicao visual `nao_avaliado`.
7. `visual.py` gera a imagem anotada, as contagens e a galeria; JSON/PNG podem ser exportados.

## Quais modulos sao usados no painel?

| Arquivo | Papel | Necessario para abrir o painel? |
|---|---|---|
| `app_alimentos.py` | Interface principal | Sim |
| `catalogo.py` | Nomes, indices e classes | Sim |
| `fluxo.py` | Orquestracao e detectores | Sim |
| `profundo.py` | Carregamento e inferencia ResNet/CNN | Sim se IA habilitada |
| `classico.py` e `classico_core.py` | Classificacao visual sem redes | Sim se classico habilitado |
| `visual.py` | Desenho, galeria e resumo | Sim |
| `models/yolo11n.pt` | Detector | Sim (ou download previo) |
| `models/alimentos_multiclasse.pth` | ResNet treinada | Sim se IA habilitada |
| `models/limiares_classico.json` | Limiares calibrados | Recomendado; ha heuristicas de fallback |
| `scripts/*.py` | Utilitarios de manutencao/treino | Nao, apenas quando necessarios |
| `tests/*.py` | Verificacoes automatizadas | Nao, apenas para desenvolver |

## Cuidados de avaliacao

- Uma deteccao nao garante que a caixa contenha um unico objeto.
- Menor `conf` pode melhorar recall de frutos deteriorados e elevar falsos positivos.
- Os resultados de testes com imagens isoladas nao medem a eficacia do pipeline com YOLO; e necessario testar as caixas em novas fotos independentes.
- O metodo classico usa classe do YOLO; o classificador de IA pode discordar.
- Fotos geradas por IA servem como demos, mas nao substituem teste com fotos reais anotadas.
- Nenhum resultado confirma se um alimento e seguro para consumo.

## Treinamento e calibração (atualização)

- `src/alimentos_v2/conjuntos.py`: importa conjuntos em pastas `train/val` ou em um manifesto CSV compatível; nunca usa `test` para treino/calibração.
- `src/alimentos_v2/treinamento.py`: ResNet-18/CNN própria, Transfer Learning, CrossEntropy ponderada por frequência da classe, AdamW, augmentation de treino, AMP/GradScaler em CUDA, Early Stopping pelo F1 macro da validação, checkpoint na melhor época.
- `scripts/treinar_multiclasse.py`: interface CLI do treinamento.
- `src/alimentos_v2/calibracao.py`: seleciona por espécie o ponto de corte do escore clássico `0.85*manchas + 0.15*textura` maximizando F1 macro da validação.
- `scripts/calibrar_classico.py`: interface CLI da calibração; cria um JSON novo, sem alterar pesos da ResNet.

O painel carrega checkpoints `.pth` antigos e novos no mesmo formato. A calibração não interfere nos pesos da rede e a inferência não utiliza rótulos do conjunto de treinamento. Validação em imagens individuais não implica desempenho equivalente em recortes de fotos com múltiplos alimentos.
