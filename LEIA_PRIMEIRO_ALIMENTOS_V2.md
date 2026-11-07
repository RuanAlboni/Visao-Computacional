# Alimentos V2 — cenoura e painel multialimento

## O que muda
Pacote aditivo: NAO substitui app.py, modulos tomato/apple ou classificador original.
Use python -m streamlit run app_alimentos.py para a nova tela.
Motor detector compartilhado: carrot, broccoli, banana, apple, orange.
CarrotModule implementa o ABC existente e usa o mesmo motor. Estado UNKNOWN sem pesos.
Nenhuma regra de cenoura fresca/podre foi inventada; este pacote nao conclui a abordagem
classica de cenoura. O fluxo classico anterior permanece separado. Nao afirmar que as
exigencias da disciplina foram todas satisfeitas.

## Instalacao
Na raiz, venv ativo:
python -m pip install -r requirements-alimentos-v2.txt
python scripts/baixar_detector_alimentos.py
python -m pytest -q tests/test_alimentos_v2.py
python -m streamlit run app_alimentos.py
O script baixar_detector_alimentos faz download explicito de yolo11n.pt oficial (COCO),
verifica nomes das classes e copia pesos para models/yolo11n.pt. Requer internet e
compatibilidade de torch/ultralytics. Verifique as licencas aplicaveis ao uso do software.
Nenhum commit e realizado. Pesos nao estao incluidos no ZIP.
Extraia com unzip -n para nao sobrescrever arquivos preexistentes de mesmo nome.

## Modelo multiclasse de Ruan
Um checkpoint no formato enviado (arquitetura, classes, modelo, tamanho) pode ser carregado
pela nova implementacao. ResNet18/CNN com a mesma arquitetura sao suportadas. Classes e
ordem sao lidas do checkpoint. Validacao strict de state_dict. Sem adivinhar nomes/camadas.
O pre-processamento padrao coincide com o codigo enviado: resize quadrado 224, media e
desvio ImageNet. Se a branch real usar outro pre-processamento/arquitetura, adequar antes.
Inspecionar:
python scripts/inspecionar_checkpoint_alimentos.py caminho/ruan.pth
Na tela, habilitar classificador e informar esse caminho. Checkpoint de maca/banana/laranja
avalia apenas esses produtos; cenoura/brocolis permanecem nao_avaliado. Discordancia de
produto entre detector e classificador produz inconclusivo. Score maximo nao e calibracao
estatistica e nao confirma seguranca. Carregar somente arquivos de origem confiavel.

## Treinar um unico ResNet para N classes produto+condicao
Estrutura:
data/alimentos/train/maca_fresca/*.jpg
data/alimentos/train/maca_deteriorada/*.jpg
... banana_fresca, banana_deteriorada, laranja_fresca, laranja_deteriorada
... cenoura_fresca, cenoura_deteriorada (somente se houver dados)
Repita as mesmas pastas em val e test com imagens independentes.
Nomes admitidos: produtos do catalogo + sufixos fresco/fresca/deteriorado/deteriorada.
Nao use dois sufixos distintos para representar a mesma classe.
Separe por fruto/lote/sequencia de captura; fotos quase iguais do mesmo item nao devem
ficar nos dois conjuntos. Nao use test para escolher thresholds ou checkpoint.
python scripts/treinar_multiclasse.py --dados data/alimentos --epocas 15 --device cpu
O melhor checkpoint e escolhido por F1 macro em val, o test permanece reservado.
Pre-treino ImageNet e baixado na primeira execucao; --sem-pretreino evita esse download,
mas nao equivale a treinamento por transfer learning. A aba treinamento gera o comando,
nao inicia jobs escondidos ou download automatico. Nao existe avaliacao final implementada
neste pacote; acrescentar avaliacao em test antes de alegar desempenho.

## Validacao feita neste ambiente
10 testes passaram em Python 3.12, sem torch/torchvision/ultralytics instalados, usando
backend simulado para testar roteamento de classes, BGR, filtro de cenoura, estado ausente,
discordancia, IDs e JSON. Todos os fontes compilados. Redes, downloads, treinamento e
interface no navegador NAO foram executados aqui. Confirmar no ambiente Python 3.14 local.
Nenhuma acuracia ou facilidade relativa de frutas foi medida.

## Relatorios
Uma submissao, um ID; PNG e JSON da mesma analise, hash da imagem e dos pesos.
Log rotativo em logs/alimentos_v2.log. Numero exibido e previsao do detector, nao verdade.
Sem prazo estimado. Ramo classico anterior continua na aplicacao anterior, nao e fundido
artificialmente com o novo detector. Estado visual fresco/podre nao equivale a seguranca.
