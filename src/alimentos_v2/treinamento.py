"""Treinamento reproduzível para as arquiteturas já aceitas pelo painel.

Mantém o formato de checkpoints do Smart Fridge Vision e acrescenta os recursos
experimentais do treinamento: augmentation, pesos de classe, AMP,
early stopping e histórico de treino/validação.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .catalogo import validar_classes
from .profundo import criar, transformar


def dispositivo_escolhido(nome: str = "auto"):
    import torch

    if nome not in {"auto", "cpu", "cuda"}:
        raise ValueError("Use auto, cpu ou cuda.")
    if nome == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitado, mas indisponível. Verifique a instalação do PyTorch/driver NVIDIA.")
    return torch.device("cuda" if nome == "cuda" or (nome == "auto" and torch.cuda.is_available()) else "cpu")


def transformacao_treino(tamanho: int = 224):
    """Augmentation somente no treino; validação/inferência sem aleatoriedade."""
    from torchvision import transforms as T

    return T.Compose([
        T.RandomResizedCrop(tamanho, scale=(0.78, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(12),
        T.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def executar_epoca(modelo, loader, dispositivo, criterio, otimizador=None, scaler=None):
    """Retorna métricas tanto para treino como validação, sem alterar pesos na validação."""
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    treinando = otimizador is not None
    modelo.train(treinando)
    exemplos = 0
    perda_total = 0.0
    reais, previstos = [], []
    with torch.set_grad_enabled(treinando):
        for imagens, rotulos in loader:
            imagens = imagens.to(dispositivo, non_blocking=True)
            rotulos = rotulos.to(dispositivo, non_blocking=True)
            with torch.autocast(device_type=dispositivo.type, dtype=torch.float16,
                                enabled=dispositivo.type == "cuda"):
                logits = modelo(imagens)
                perda = criterio(logits, rotulos)
            if treinando:
                otimizador.zero_grad(set_to_none=True)
                scaler.scale(perda).backward()
                scaler.step(otimizador)
                scaler.update()
            perda_total += perda.detach().item() * len(rotulos)
            exemplos += len(rotulos)
            reais.extend(rotulos.detach().cpu().tolist())
            previstos.extend(logits.argmax(dim=1).detach().cpu().tolist())

    if exemplos == 0:
        raise ValueError("Conjunto vazio durante a execução de uma época.")
    return {
        "perda": perda_total / exemplos,
        "acuracia": float(accuracy_score(reais, previstos)),
        "f1_macro": float(f1_score(reais, previstos, labels=list(range(len(loader.dataset.classes))),
                                    average="macro", zero_division=0)),
    }


def treinar(
    dados: str | Path,
    saida: str | Path = "models/alimentos_multiclasse.pth",
    arquitetura: str = "resnet18",
    epocas: int = 25,
    lote: int = 32,
    paciencia: int = 6,
    taxa: float | None = None,
    sem_pretreino: bool = False,
    dispositivo: str = "auto",
    workers: int = 0,
    semente: int = 42,
) -> Path:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from .conjuntos import datasets_treino_validacao

    if arquitetura not in {"resnet18", "cnn"}:
        raise ValueError("Arquitetura deve ser resnet18 ou cnn.")
    if min(epocas, lote, paciencia) < 1 or workers < 0:
        raise ValueError("Épocas, lote e paciência devem ser positivos; workers não pode ser negativo.")
    if taxa is not None and not 0 < taxa < 1:
        raise ValueError("A taxa de aprendizado deve estar entre 0 e 1 (exclusive).")

    dev = dispositivo_escolhido(dispositivo)
    torch.manual_seed(semente)
    np.random.seed(semente)
    random.seed(semente)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(semente)
        torch.backends.cudnn.benchmark = True  # otimiza desempenho; não garante determinismo total

    treino, validacao = datasets_treino_validacao(dados, transformacao_treino(), transformar(False))
    classes = validar_classes(treino.classes)
    if treino.class_to_idx != validacao.class_to_idx:
        raise ValueError("As classes e os índices devem ser idênticos em train e val.")
    if not treino.samples or not validacao.samples:
        raise ValueError("Os conjuntos de treino e validação não podem estar vazios.")
    caminhos_treino = {p for p, _ in treino.samples}
    caminhos_val = {p for p, _ in validacao.samples}
    if caminhos_treino & caminhos_val:
        raise ValueError("A mesma imagem aparece em treino e validação; corrija os splits.")
    quantidades = np.bincount([classe for _, classe in treino.samples], minlength=len(classes))
    quant_val = np.bincount([classe for _, classe in validacao.samples], minlength=len(classes))
    if (quantidades == 0).any() or (quant_val == 0).any():
        raise ValueError("Todas as classes devem ter pelo menos uma imagem em train e val.")

    argumentos_loader = {"batch_size": lote, "num_workers": workers,
                         "pin_memory": dev.type == "cuda"}
    if workers > 0:
        argumentos_loader["persistent_workers"] = True
    train_loader = DataLoader(treino, shuffle=True, **argumentos_loader)
    val_loader = DataLoader(validacao, shuffle=False, **argumentos_loader)

    modelo = criar(arquitetura, len(classes),
                   pretreinado=(arquitetura == "resnet18" and not sem_pretreino)).to(dev)
    pesos = torch.tensor(len(treino) / (len(classes) * quantidades),
                         dtype=torch.float32, device=dev)
    criterio = nn.CrossEntropyLoss(weight=pesos)
    taxa = taxa if taxa is not None else (1e-4 if arquitetura == "resnet18" else 1e-3)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=taxa, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    destino = Path(saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    melhor, sem_melhora, historico = -1.0, 0, []
    print(f"Dispositivo: {dev} | GPU: {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'não utilizada'}")
    print(f"Arquitetura: {arquitetura} | classes: {', '.join(classes)} | treino: {len(treino)} | validação: {len(validacao)}")

    for epoca in range(1, epocas + 1):
        met_t = executar_epoca(modelo, train_loader, dev, criterio, otimizador, scaler)
        met_v = executar_epoca(modelo, val_loader, dev, criterio)
        linha = {"epoca": epoca, "train": met_t, "val": met_v}
        historico.append(linha)
        print(f"Época {epoca:02d}/{epocas} | treino perda {met_t['perda']:.4f}, F1 {met_t['f1_macro']:.4f} "
              f"| validação perda {met_v['perda']:.4f}, F1 {met_v['f1_macro']:.4f}", flush=True)
        if met_v["f1_macro"] > melhor + 1e-5:
            melhor, sem_melhora = met_v["f1_macro"], 0
            # Formato compatível com src/alimentos_v2/profundo.py::Classificador.
            torch.save({
                "arquitetura": arquitetura, "classes": list(classes),
                "modelo": modelo.state_dict(), "tamanho": 224,
                "media": [0.485, 0.456, 0.406], "desvio": [0.229, 0.224, 0.225],
                "epoca": epoca, "f1_validacao": melhor, "historico": historico,
            }, destino)
            print(f"  Melhor checkpoint salvo: {destino}")
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                print(f"Early stopping: {paciencia} épocas consecutivas sem melhora.")
                break

    destino.with_suffix(".historico.json").write_text(
        json.dumps(historico, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Treinamento concluído. Melhor F1 macro (validação): {melhor:.4f}")
    print("Teste independente não foi usado neste treinamento; avalie posteriormente.")
    return destino
