# Smart Fridge Vision — painel multialimento

Aplicacao local em Streamlit que localiza alimentos usando **YOLO11n** e compara **ResNet-18** e **visao computacional classica** para avaliar sinais visuais de deterioracao em **maca, banana e laranja**. Para **cenoura** e **brocolis**, apenas detecta; o estado nao e avaliado.

> **Limite importante:** fresco/deteriorado e uma previsao visual; a ferramenta NAO certifica seguranca alimentar.

## Organizacao

```text
SmartFridgeVision/
├── app_alimentos.py                 # unica interface Streamlit ativa
├── src/
│   └── alimentos_v2/
│       ├── catalogo.py              # alimentos e rótulos de estado
│       ├── fluxo.py                 # YOLO -> recortes -> classificadores
│       ├── classico.py              # adaptador para os recortes do YOLO
│       ├── classico_core.py         # HSV/LAB, morfologia, LBP/GLCM, regras
│       ├── profundo.py              # ResNet/CNN e inferencia de checkpoints
│       ├── conjuntos.py             # pastas train/val ou manifesto CSV
│       ├── treinamento.py           # treinamento profundo aprimorado
│       ├── calibracao.py            # busca de limiares de validação
│       └── visual.py                # overlays, galeria e resumo
├── models/
│   ├── yolo11n.pt                   # detector
│   ├── alimentos_multiclasse.pth    # ResNet-18 treinada: 6 classes
│   └── limiares_classico.json       # calibracao do metodo classico
├── scripts/
│   ├── baixar_detector_alimentos.py
│   ├── inspecionar_checkpoint_alimentos.py
│   ├── treinar_multiclasse.py       # treinamento com Early Stopping e AMP
│   └── calibrar_classico.py        # calibra limiares sem retreinar a IA
├── tests/
│   ├── test_alimentos_v2.py
│   ├── test_classico_integrado.py
│   └── test_treinamento_calibracao.py
├── docs/
│   ├── ARQUITETURA.md
│   └── LIMPEZA.md
├── requirements.txt
├── .gitignore
└── .gitattributes                  # Git LFS nos checkpoints
```

## Instalacao no Windows (Python 3.11 recomendado)

No PowerShell, na pasta do projeto:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**Com CPU apenas** (escolha uma das variantes, nao as duas):

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Com GPU NVIDIA compatível:** use o comando apropriado para o driver e GPU em <https://pytorch.org/get-started/locally/>. Em seguida, para ambos os ambientes:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app_alimentos.py
```

Selecione `cpu` na interface se nao tiver CUDA; uma GPU NVIDIA compativel pode selecionar `cuda`. A instalacao da versao CPU funciona para inferencia com os mesmos pesos `.pth`, sem novo treinamento. O computador precisa ter memoria RAM suficiente para carregar modelos e imagens.

### Pesos e Git LFS

Esta distribuicao inclui `models/yolo11n.pt`, `models/alimentos_multiclasse.pth` e `models/limiares_classico.json`, mas ao publicar no GitHub **use Git LFS** para os arquivos `.pt` e `.pth`. Para clonar o repositorio versionado via LFS:

```powershell
git lfs install
git clone -b NOME_DA_BRANCH https://github.com/RuanAlboni/Visao-Computacional.git
cd Visao-Computacional
git lfs pull
```

Se obtiver o codigo sem os pesos, execute `python scripts/baixar_detector_alimentos.py` para baixar o detector oficial (requer internet) e obtenha o checkpoint treinado `alimentos_multiclasse.pth` separadamente. O detector oficial nao substitui a ResNet.

## Utilizacao

1. Execute `python -m streamlit run app_alimentos.py`.
2. Carregue uma imagem (`jpg`, `png` ou `webp`).
3. Selecione os alimentos (ou analise todos).
4. Ajuste o score minimo do detector se frutas deterioradas deixarem de ser detectadas. **Reduzir o limiar aumenta o risco de falsos positivos.**
5. Habilite `Usar classificador IA` para carregar a ResNet treinada; mantenha `Usar classificador clássico` habilitado para comparar as duas abordagens.
6. Veja as caixas, recortes, estados, pontuacoes e divergencias. Exporte PNG/JSON, se desejar.

**As deteccoes do YOLO sao a entrada dos classificadores:** se o YOLO nao detectar um objeto, ele nao sera avaliado. A especie usada pelas regras classicas e a do YOLO; se ela estiver errada, a regra aplicada tambem pode estar errada. A ResNet pode sinalizar discordancia sem sobrescrever automaticamente a classe do YOLO. Os limiares classicos foram calibrados usando imagens individuais e podem nao ser adequados a fotos multialimento.

## Testes

```powershell
python -m pip install pytest
python -m pytest -q tests/
```

Os testes atuais verificam o fluxo e o metodo classico usando dados sinteticos/detectores simulados. **Nao** substituem avaliacao de acuracia, recall, mAP e erros com fotos anotadas reais.

## Treinamento opcional (atualizado)

O modelo distribuído em `models/alimentos_multiclasse.pth` **já está treinado e permanece inalterado**. O script agora utiliza um treinamento completo com: ResNet-18 com Transfer Learning, pesos de classe, data augmentation (crop, flip, rotação e ColorJitter), AdamW, FP16/AMP com CUDA, Early Stopping por F1 macro de validação, melhor checkpoint e histórico `.historico.json`. Também suporta a CNN própria (`--arquitetura cnn`).

**Formato A — diretórios `ImageFolder`:**

```text
data/alimentos/
├── train/
│   ├── maca_fresca/
│   ├── maca_deteriorada/
│   ├── banana_fresca/
│   ├── banana_deteriorada/
│   ├── laranja_fresca/
│   └── laranja_deteriorada/
└── val/
    └── (mesmas seis subpastas, com imagens diferentes)
```

As classes devem ter pelo menos uma imagem de **cada classe** em treino e validação. `test/` é opcional nesta etapa e deve permanecer reservado para a avaliação final.

**Formato B — manifesto CSV com metadados:** um diretório `dados/` contendo `manifesto.csv` e `metadados.json`. O caminho original do dataset, salvo em `metadados.json`, precisa continuar válido. As divisões `treino` e `validacao` são lidas sem copiá-las ou recomputar o split.

**Treinar usando pastas, sem sobrescrever os pesos atuais:**

```powershell
python scripts/treinar_multiclasse.py --dados data/alimentos --saida models/novo_multiclasse.pth --arquitetura resnet18 --epocas 50 --lote 32 --paciencia 6 --device cuda
```

**Treinar usando o manifesto original (exemplo):**

```powershell
python scripts/treinar_multiclasse.py --dados dados --saida models/novo_multiclasse.pth --arquitetura resnet18 --epocas 50 --lote 32 --paciencia 6 --device auto
```

Para a CNN: `--arquitetura cnn`. Para não baixar pesos ImageNet: `--sem-pretreino`. Para ajustar: `--taxa 0.0001`, `--workers 0`, `--semente 42`. `--device` aceita `auto`, `cpu` ou `cuda`; em computadores sem GPU, `auto` utiliza a CPU. O checkpoint é compatível com `Classificador` em `src/alimentos_v2/profundo.py`; basta apontar o caminho na interface para testá-lo.

**Importante:** evitar imagens duplicadas ou aumentadas de uma mesma foto em treino e validação. O script protege apenas contra o mesmo caminho literal nas duas divisões; não detecta imagens quase idênticas.

## Calibração do classificador clássico (novo)

O método clássico usa `pontuacao = 0,85 × fração_manchas + 0,15 × fração_textura`. A calibração **não treina uma rede nem muda essas regras**: seleciona um limiar por espécie (maçã, banana e laranja) maximizando F1 macro usando **somente `val/` ou `validacao` do manifesto**. Cenoura e brócolis seguem como não avaliados pelo método clássico.

```powershell
python scripts/calibrar_classico.py --dados data/alimentos --saida models/limiares_classico_novos.json --csv resultados/calibracao_classica.csv
```

Ou, com um dataset organizado em manifesto CSV:

```powershell
python scripts/calibrar_classico.py --dados dados --saida models/limiares_classico_novos.json
```

O novo JSON tem o formato já aceito pelo painel (`limiares`, `f1_validacao`, `amostras`, `falhas_segmentacao`). Por segurança, **a calibração distribuída `models/limiares_classico.json` não é sobrescrita**. Para usar o novo arquivo, informe `models/limiares_classico_novos.json` no campo *Arquivo de calibração clássico* da interface.

O F1 obtido nessa etapa mede **validação**, não teste independente. Se a calibração usar fotos isoladas e a aplicação for usada com recortes do YOLO, os melhores limiares podem mudar; para calibrar esse cenário, inclua recortes do YOLO rotulados em um conjunto de validação próprio, sem compartilhar fotografias com treino e teste.

## Reprodutibilidade e testes

```powershell
python -m pip install pytest
python -m pytest -q tests/
```

Os testes incluem checagem do formato de checkpoint, Early Stopping, suporte ao manifesto e geração de limiares. Esses testes não substituem uma avaliação com fotos reais e anotações verificadas.

## Documentacao

- [Arquitetura e interpretacao](docs/ARQUITETURA.md)

Creditos de terceiros: o detector YOLO e baseado em implementacao/modelos Ultralytics; consulte a licenca aplicavel ao seu uso e redistribuicao. Os modelos e os datasets de treinamento podem ter condicoes de uso proprias.
