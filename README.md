# 🍎 FruitVision — classificação visual de frutas

Aplicação acadêmica para comparar **visão computacional clássica baseada em regras** e **aprendizado profundo**. Escopo: **maçã, banana e laranja**, cada uma em duas condições visuais: **fresca** e **deteriorada**.

Inclui:

- Classificador clássico: segmentação, cores HSV/CIELAB, manchas, irregularidades locais de textura, LBP, GLCM e limiares por espécie.
- **CNN própria** em PyTorch, treinada do zero.
- **ResNet-18** com fine-tuning de pesos ImageNet (transfer learning).
- Importação de dataset, validação, treinamento local com CUDA/AMP e early stopping.
- Avaliação com acurácia, precisão, revocação, F1 e matriz de confusão, por modelo e por espécie.
- Aplicação **Streamlit** com classificação de novas imagens e gráficos de avaliação.

> **Importante:** as CNNs são fornecidas como **código treinável**, sem pesos treinados no dataset. É necessário baixar os dados e executar o treinamento para obter classificações de IA úteis. Os limiares clássicos iniciais também são heurísticos e devem ser calibrados. Não há resultados reais de acurácia incluídos ou alegados.

## 1. Requisitos

- Python **3.11 ou 3.12**, Windows ou Linux.
- NVIDIA RTX 5070 12 GB é adequada ao treinamento local; também é possível usar CPU (bem mais lento).
- Driver NVIDIA atualizado e conexão à internet para instalar pacotes, baixar o dataset e, no primeiro treinamento com ResNet-18, baixar pesos ImageNet.
- Não é obrigatório instalar o CUDA Toolkit separadamente quando se usam os wheels oficiais CUDA do PyTorch. Instale o wheel compatível e verifique o driver.

### Windows (PowerShell)

```powershell
cd FruitVision
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128 (para usar GPU)
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu (para usar CPU)
python -m pip install -r requirements.txt
```

Se o PowerShell bloquear scripts, use `cmd` e `.venv\Scripts\activate.bat`, ou ajuste apenas a política para o processo atual conforme as políticas do seu computador.

### Linux

```bash
cd FruitVision
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Verifique se o CUDA aparece:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

A versão e o comando CUDA são baseados na [documentação oficial do PyTorch](https://pytorch.org/get-started/previous-versions/). Em caso de instalação diferente, use o [seletor oficial](https://pytorch.org/get-started/locally/) e confira suporte à arquitetura Blackwell/`sm_120`.

## 2. Dataset

**Fruits fresh and rotten for classification** — Sriram Reddy Kalluri, Kaggle:
https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

A licença é informada pelo Kaggle como **Unknown**. Verifique as condições de uso antes de redistribuir o dataset ou publicar imagens dele. O projeto **não inclui** fotos do Kaggle.

Baixe manualmente e extraia o ZIP em uma pasta, por exemplo `C:\datasets\frutas\`. O programa procura automaticamente pastas aninhadas, como:

```text
C:\datasets\frutas\dataset\train\
    freshapples\
    freshbanana\
    freshoranges\
    rottenapples\
    rottenbanana\
    rottenoranges\
C:\datasets\frutas\dataset\test\
    (as mesmas seis pastas)
```

Alternativamente, após configurar a API do Kaggle:

```bash
pip install kaggle
kaggle datasets download -d sriramr/fruits-fresh-and-rotten-for-classification -p downloads --unzip
```

**Atenção à integridade experimental:** o dataset contém imagens aumentadas (rotações, recortes etc.). A divisão treino/validação agrupa nomes de arquivos parecidos quando possível, mas **isso não garante ausência de vazamento**. Fotos quase idênticas podem ter nomes diferentes e versões aumentadas podem existir inclusive no teste original. Antes de reportar resultados como generalização, realize auditoria visual/perceptual das duplicatas e, se possível, crie grupos por fotografia original (não por arquivo aumentado). É uma limitação conhecida deste baseline.

## 3. Preparar os dados

O comando usa as imagens no local original (não faz cópias gigantes). Preserva o teste fornecido e cria validação **somente a partir do treino**.

```powershell
python main.py preparar --origem "C:\datasets\frutas" --saida dados --validacao 0.20 --semente 42
```

Para um teste rápido usando amostra de até 10 imagens por classe e por divisão:

```powershell
python main.py preparar --origem "C:\datasets\frutas" --limite-por-classe 10
```

O manifesto fica em `dados/manifesto.csv`, com metadados em `dados/metadados.json`. Se o dataset mudar de lugar, execute novamente `preparar`.

## 4. Calibrar o classificador clássico

```bash
python main.py calibrar --dados dados
```

Os limiares são otimizados **apenas com o conjunto de validação** para F1-macro binário, separadamente para maçã, banana e laranja, e salvos em `modelos/limiares_classico.json`. A segmentação e as regras continuam explicitamente programadas; não há rede neural nessa etapa. LBP e GLCM são calculados como indicadores descritivos; a decisão utiliza a proporção de regiões com manchas e textura irregular.

> Fotos com fundos complexos podem falhar na segmentação. O algoritmo informa `indeterminado` em vez de inventar uma resposta; a avaliação inclui a cobertura (percentual de imagens em que conseguiu decidir). As regras podem precisar de ajustes após examinar amostras reais do dataset.

## 5. Treinar as redes neurais (RTX 5070)

Comece com ResNet-18:

```bash
python main.py treinar --dados dados --arquitetura resnet18 --epocas 25 --lote 32 --dispositivo cuda
```

Depois, opcionalmente, treine uma CNN própria do zero:

```bash
python main.py treinar --dados dados --arquitetura cnn --epocas 35 --lote 32 --dispositivo cuda
```

Os arquivos são salvos como `modelos/resnet18.pth` e `modelos/cnn.pth`, com históricos `.json`. O melhor checkpoint é escolhido pelo F1-macro das seis classes na **validação**. Há data augmentation de treinamento, pesos de classe, otimizador AdamW, AMP em CUDA e early stopping.

Se houver erro de memória, use `--lote 16` ou `--lote 8`. No Windows, deixe `--workers 0` inicialmente. Para treinar a ResNet sem baixar pesos externos, adicione `--sem-pretreino` (o que muda a natureza do experimento). `--dispositivo cuda` falha explicitamente se CUDA não estiver disponível; `auto` usa CPU como alternativa.

## 6. Avaliar no teste reservado

**Execute só depois** de calibrar e treinar, para evitar usar o teste na escolha de parâmetros:

```bash
python main.py avaliar --dados dados --pesos modelos/resnet18.pth modelos/cnn.pth --saida resultados
```

Se quiser avaliar apenas o método clássico:

```bash
python main.py avaliar --dados dados --saida resultados
```

São gerados `resultados/resumo.json`, CSVs de predições, matrizes de confusão em PNG, métricas binárias gerais e por espécie, e métricas de seis classes para cada CNN. A parte clássica usa a espécie verdadeira fornecida pelo dataset; as CNNs precisam prever a espécie e o estado simultaneamente. Essa diferença deve ser declarada no relatório.

## 7. Executar a aplicação

```bash
streamlit run app.py
```

Abra o endereço local mostrado pelo Streamlit (normalmente http://localhost:8501). Na aba **Classificação**, envie uma imagem e selecione a espécie para as regras clássicas. Se existirem pesos treinados em `modelos/`, o app executa também as CNNs. Em **Experimentos**, veja métricas e gráficos reais gerados pelo passo anterior.

A aplicação não baixa imagens nem envia fotografias a serviços externos para inferência. A primeira execução do treinamento ResNet-18 com pesos pré-treinados pode baixá-los do site do PyTorch.

## 8. Testes técnicos

```bash
python -m pip install pytest
python -m pytest -q
```

Os testes usam imagens **artificiais** e verificam a interface dos algoritmos, checkpoints e a criação de manifestos. **Não validam acurácia em fotografias reais.**

## Estrutura

```text
FruitVision/
  app.py                     # interface Streamlit
  main.py                    # comandos preparar/calibrar/treinar/avaliar
  src/
    classico.py              # segmentação, manchas, textura, regras
    profundo.py              # CNN, ResNet-18, treino e inferência
    dados.py                 # manifesto estratificado por grupos
    avaliacao.py             # calibração e métricas
    constantes.py            # classes e nomes das frutas
  tests/test_projeto.py
  exemplos/                  # imagens artificiais de teste da interface
  modelos/                   # checkpoints produzidos ao treinar
  resultados/                # relatórios após avaliar
  requirements.txt
```

## Limites e validade

1. **Frescor visual não prova comestibilidade.** Contaminação pode não ser perceptível na fotografia, e manchas podem não indicar risco. O sistema não deve ser usado para decidir segurança alimentar real.
2. **Limiar clássico e fundo:** a segmentação por cores/GrabCut supõe uma fruta destacável do fundo. É esperado que imagens complexas causem erros.
3. **Distribuição e vazamento:** imagens artificialmente aumentadas e condições uniformes podem inflar métricas. Faça avaliação adicional com fotografias externas de câmeras e cenários distintos.
4. **Comparação:** o classificador clássico recebe a espécie manualmente, mas as CNNs aprendem espécie + condição. Compare sobretudo o estado visual e registre a diferença de informação disponível.
5. **Probabilidades da IA:** são saídas softmax, não garantem confiança calibrada.

Projeto acadêmico e demonstrativo, não ferramenta de inspeção sanitária.
