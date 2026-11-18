# Atualização — treinamento profundo e calibração clássica

## O que mudou

- Reescrito `scripts/treinar_multiclasse.py`, agora conectado a `src/alimentos_v2/treinamento.py`.
- Recursos do treinamento: early stopping (F1 macro), mixed precision CUDA, balanceamento por classe, augmentation, checkpoint de melhor época e histórico.
- Aceita `ImageFolder` em `train/val` **ou** `dados/manifesto.csv` e `dados/metadados.json` para datasets organizados por manifesto CSV.
- Mantida inferência e compatibilidade com checkpoints existentes de ResNet-18/CNN.
- Criado `scripts/calibrar_classico.py`, que calcula limiares de cor/textura em `val` e não mexe na rede neural.
- Nenhum checkpoint antigo foi substituído no pacote.

## Comandos

```powershell
# Treinar ResNet-18 sem sobrescrever o checkpoint distribuído:
python scripts/treinar_multiclasse.py --dados dados --saida models/novo_multiclasse.pth --arquitetura resnet18 --device cuda --epocas 50 --lote 32 --paciencia 6

# Calibrar usando apenas validação (pastas ou manifesto):
python scripts/calibrar_classico.py --dados dados --saida models/limiares_classico_novos.json --csv resultados/pontuacoes_validacao.csv

# Abrir a aplicação (os checkpoints distribuídos continuam funcionando):
python -m streamlit run app_alimentos.py
```

Substitua `dados` por `data/alimentos` se utilizar a organização em pastas. Se usar o manifesto, garanta que a pasta de origem do dataset ainda existe no caminho indicado pelos metadados.

## Cautelas experimentais

- Os pesos antigos não precisam ser retreinados para executar a aplicação.
- O melhor checkpoint é o que maximiza F1 macro na validação, não necessariamente o da última época.
- O limite clássico é um parâmetro ajustado no conjunto de **validação**, não uma probabilidade de segurança alimentar.
- Não use o conjunto `test` na calibração ou para selecionar a melhor época.
- Não coloque imagens aumentadas de uma mesma foto nos dois lados de uma divisão.
- Para inferência com recortes YOLO, é ideal calibrar e medir desempenho em recortes YOLO rotulados, independentes.
- Não treine e avalie apenas com imagens geradas ou artificiais.
