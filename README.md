# 🍓 Smart Fridge Vision

Sistema de visão computacional para monitoramento de frescor de frutas em geladeiras inteligentes.

## 🎯 Objetivo

Classificar **tomate**, **maçã** e **morango** como:
- ✅ **APPROPRIATE** (próprio para consumo)
- ❌ **INAPPROPRIATE** (impróprio para consumo)

## 🛠️ Tecnologias

- **YOLOv8**: Detecção e classificação
- **OpenCV**: Processamento de imagens
- **Python 3.10+**

## 📁 Estrutura
smart_fridge_vision/
│
├── 📄 README.md # Documentação principal
├── 📄 requirements.txt # Dependências Python
├── 📄 .gitignore # Arquivos ignorados pelo Git
├── 📄 .env.example # Exemplo de variáveis de ambiente
│
├── 📂 config/ # Configurações do projeto
│ ├── _init_.py
│ ├── settings.py # Configurações globais (paths, thresholds)
│ └── products.py # Config específico por produto
│
├── 📂 data/ # Datasets
│ ├── 📂 raw/ # Datasets brutos (não versionar)
│ │ ├── tomato/ # Dataset de tomate
│ │ │ ├── images/
│ │ │ │ ├── train/
│ │ │ │ ├── val/
│ │ │ │ └── test/
│ │ │ └── labels/
│ │ │ ├── train/
│ │ │ ├── val/
│ │ │ └── test/
│ │ ├── apple/ # Dataset de maçã
│ │ └── strawberry/ # Dataset de morango
│ │
│ ├── 📂 processed/ # Datasets pré-processados
│ │ ├── yolo_format/
│ │ └── svm_format/
│ │
│ └── 📂 splits/ # Train/val/test splits
│ ├── train.txt
│ ├── val.txt
│ └── test.txt
│
├── 📂 models/ # Modelos treinados (não versionar)
│ ├── 📂 yolo/ # Modelos YOLO
│ │ ├── tomato_v1.pt
│ │ ├── apple_v1.pt
│ │ └── strawberry_v1.pt
│ │
│ └── 📂 svm/ # Modelos SVM (abordagem clássica)
│ ├── tomato_v1.pkl
│ ├── apple_v1.pkl
│ └── strawberry_v1.pkl
│
├── 📂 src/ # Código fonte
│ ├── _init_.py
│ │
│ ├── 📂 data/ # Manipulação de dados
│ │ ├── _init_.py
│ │ ├── dataset_loader.py # Carregar datasets
│ │ ├── preprocessor.py # Pré-processamento (resize, normalize)
│ │ └── augmentator.py # Data augmentation
│ │
│ ├── 📂 features/ # Extração de features (SVM)
│ │ ├── _init_.py
│ │ ├── color_features.py # HSV histograms
│ │ ├── texture_features.py # LBP
│ │ └── shape_features.py # Hu moments
│ │
│ ├── 📂 models/ # Modelos de ML/DL
│ │ ├── _init_.py
│ │ ├── yolo_model.py # YOLOv8 training/inference
│ │ └── svm_model.py # SVM training/inference
│ │
│ ├── 📂 utils/ # Utilitários
│ │ ├── _init_.py
│ │ ├── visualization.py # Plot bounding boxes, results
│ │ ├── metrics.py # Accuracy, precision, recall, mAP
│ │ └── file_utils.py # Save/load models, images
│ │
│ └── 📂 modules/ # Módulos por produto
│ ├── _init_.py
│ │
│ ├── 📂 tomato/ # Módulo de tomate
│ │ ├── _init_.py
│ │ ├── config.py # Config específico
│ │ ├── train.py # Treino específico
│ │ └── predict.py # Inferência específica
│ │
│ ├── 📂 apple/ # Módulo de maçã
│ │ ├── _init_.py
│ │ ├── config.py
│ │ ├── train.py
│ │ └── predict.py
│ │
│ └── 📂 strawberry/ # Módulo de morango
│ ├── _init_.py
│ ├── config.py
│ ├── train.py
│ └── predict.py
│
├── 📂 notebooks/ # Jupyter notebooks para experimentos
│ ├── 01_eda_tomato.ipynb
│ ├── 02_eda_apple.ipynb
│ ├── 03_eda_strawberry.ipynb
│ └── 04_model_comparison.ipynb
│
├── 📂 scripts/ # Scripts de linha de comando
│ ├── download_datasets.sh
│ ├── preprocess_data.py
│ ├── train_yolo.py
│ ├── train_svm.py
│ └── run_inference.py
│
├── 📂 tests/ # Testes unitários
│ ├── _init_.py
│ ├── test_data_loader.py
│ ├── test_yolo_model.py
│ └── test_svm_model.py
│
├── 📂 demos/ # Demos e exemplos
│ ├── demo_single_image.py
│ ├── demo_batch.py
│ └── demo_smart_fridge.py
│
├── 📂 runs/ # Resultados de treino (não versionar)
│ └── 📂 detect/
│ └── 📂 models/
│ └── 📂 yolo/
│ └── tomato_v1-5/
│ ├── weights/
│ │ ├── best.pt
│ │ └── last.pt
│ ├── results.csv
│ └── confusion_matrix.png
│
└── 📂 reports/ # Relatórios e apresentações
├── 📂 figures/
├── 📂 tables/
└── presentation_preliminary.pptx


## 📊 Status do Projeto

### ✅ Implementado

- [x] Estrutura de diretórios modular
- [x] Setup do ambiente (WSL2, Python, YOLOv8)
- [x] Pipeline de treino YOLOv8 funcional
- [x] Dataset COCO128 (exemplo)
- [x] Script de inferência básico
- [x] Demo funcional
- [x] Git configurado com branches por produto

### 🚧 Em Desenvolvimento

- [ ] Dataset específico de tomate (apropriado vs impróprio)
- [ ] Dataset específico de maçã
- [ ] Dataset específico de morango
- [ ] Treino de modelos específicos
- [ ] Estimativa de dias restantes
- [ ] Integração dos módulos

### 📅 Próximos Passos

- [ ] Coleta de datasets específicos
- [ ] Treino e validação dos modelos
- [ ] Comparação YOLOv8 vs SVM
- [ ] Implementação de estimativa temporal
- [ ] Deploy em hardware embarcado

---

## 🗺️ Roadmap

### Fase 1: Setup (Setembro 2026) ✅

- [x] Estrutura do projeto
- [x] Ambiente configurado
- [x] YOLOv8 treinado (COCO128)
- [x] Demo funcional

### Fase 2: Datasets (Outubro 2026)

- [ ] Dataset de tomate (apropriado vs impróprio)
- [ ] Dataset de maçã
- [ ] Dataset de morango
- [ ] Validação dos datasets

### Fase 3: Modelos (Novembro 2026)

- [ ] Treino modelos específicos
- [ ] Comparação YOLOv8 vs SVM
- [ ] Estimativa de dias restantes
- [ ] Integração dos módulos

### Fase 4: Deploy (Dezembro 2026)

- [ ] Otimização para edge (Raspberry Pi)
- [ ] Interface de usuário
- [ ] Documentação final
- [ ] Apresentação

Criar Ambiente Virtual

```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Instalar Dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verificar Instalação

```bash
python -c "import ultralytics; print(f'YOLOv8: {ultralytics.__version__}')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### Rodar Demo

```bash
# Demo completa
python demos/demo_smart_fridge.py
```