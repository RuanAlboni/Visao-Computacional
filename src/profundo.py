"""Treinamento/inferencia de CNN propria e ResNet-18 (6 classes)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import f1_score, accuracy_score
from .constantes import CLASSES, decompor_classe
from .dados import carregar_manifesto

TAMANHO = 224
MEDIA = [0.485, 0.456, 0.406]
DESVIO = [0.229, 0.224, 0.225]


class CNNPequena(nn.Module):
    """CNN compacta treinada sem transfer learning."""
    def __init__(self, saidas: int = 6):
        super().__init__()
        blocos = []
        canais = [3, 32, 64, 128, 192]
        for anterior, atual in zip(canais[:-1], canais[1:]):
            blocos += [nn.Conv2d(anterior, atual, 3, padding=1, bias=False),
                       nn.BatchNorm2d(atual), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        self.extrator = nn.Sequential(*blocos, nn.AdaptiveAvgPool2d(1))
        self.classificador = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(canais[-1], saidas))

    def forward(self, x):
        return self.classificador(self.extrator(x))


def criar_modelo(arquitetura: str, pretreinado: bool = False) -> nn.Module:
    if arquitetura == "cnn":
        return CNNPequena(saidas=len(CLASSES))
    if arquitetura == "resnet18":
        pesos = models.ResNet18_Weights.DEFAULT if pretreinado else None
        modelo = models.resnet18(weights=pesos)
        modelo.fc = nn.Linear(modelo.fc.in_features, len(CLASSES))
        return modelo
    raise ValueError("Arquitetura deve ser 'cnn' ou 'resnet18'.")


def transformacao(treino: bool = False):
    etapas = ([transforms.RandomResizedCrop(TAMANHO, scale=(0.78, 1.0)),
               transforms.RandomHorizontalFlip(),
               transforms.RandomRotation(12),
               transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.12)] if treino else
              [transforms.Resize((TAMANHO, TAMANHO))])
    return transforms.Compose(etapas + [transforms.ToTensor(), transforms.Normalize(MEDIA, DESVIO)])


class ImagensDataset(Dataset):
    def __init__(self, linhas, base: Path, treino: bool = False):
        self.linhas = linhas.reset_index(drop=True)
        self.base = base
        self.transformar = transformacao(treino)

    def __len__(self):
        return len(self.linhas)

    def __getitem__(self, indice):
        linha = self.linhas.iloc[indice]
        caminho = self.base / linha.arquivo
        with Image.open(caminho) as origem:
            img = ImageOps.exif_transpose(origem).convert("RGB")
        img = self.transformar(img)
        return img, CLASSES.index(linha.classe)


def dispositivo_escolhido(nome: str = "auto") -> torch.device:
    if nome not in {"auto", "cuda", "cpu"}:
        raise ValueError("Use auto, cuda ou cpu.")
    if nome == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA solicitado mas indisponivel. Verifique o PyTorch e o driver NVIDIA.")
    return torch.device("cuda" if nome == "cuda" or (nome == "auto" and torch.cuda.is_available()) else "cpu")


def _lote(tamanho: int, frame, base: Path, treino: bool, workers: int, device: torch.device):
    dataset = ImagensDataset(frame, base, treino=treino)
    return DataLoader(dataset, batch_size=tamanho, shuffle=treino, num_workers=workers,
                      pin_memory=(device.type == "cuda"), persistent_workers=workers > 0)


def _epoca(modelo, loader, dispositivo, criterion, optimizer=None, scaler=None):
    treinando = optimizer is not None
    modelo.train(treinando)
    soma_perda, exemplos = 0.0, 0
    certos, reais = [], []
    for imagens, labels in loader:
        imagens = imagens.to(dispositivo, non_blocking=True)
        labels = labels.to(dispositivo, non_blocking=True)
        with torch.set_grad_enabled(treinando):
            with torch.autocast(device_type=dispositivo.type, dtype=torch.float16,
                                enabled=dispositivo.type == "cuda"):
                logits = modelo(imagens)
                perda = criterion(logits, labels)
            if treinando:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(perda).backward()
                scaler.step(optimizer)
                scaler.update()
        soma_perda += float(perda.item()) * len(labels)
        exemplos += len(labels)
        certos.extend(logits.argmax(dim=1).detach().cpu().tolist())
        reais.extend(labels.detach().cpu().tolist())
    return {"perda": soma_perda / max(1, exemplos),
            "acuracia": accuracy_score(reais, certos),
            "f1_macro": f1_score(reais, certos, labels=list(range(len(CLASSES))), average="macro", zero_division=0)}


def treinar(dados: str = "dados", arquitetura: str = "resnet18", saida: str = "modelos/resnet18.pth",
            epocas: int = 25, lote: int = 32, paciencia: int = 6, taxa: float | None = None,
            sem_pretreino: bool = False, dispositivo: str = "auto", workers: int = 0, semente: int = 42):
    frame, base = carregar_manifesto(dados)
    dev = dispositivo_escolhido(dispositivo)
    print(f"Dispositivo: {dev} | GPU: {torch.cuda.get_device_name(0) if dev.type == 'cuda' else 'nao utilizada'}")
    torch.manual_seed(semente)
    np.random.seed(semente)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(semente)
        torch.backends.cudnn.benchmark = True  # mais rapido; nao garante determinismo completo
    modelo = criar_modelo(arquitetura, pretreinado=(arquitetura == "resnet18" and not sem_pretreino)).to(dev)
    treino_frame = frame[frame.divisao == "treino"]
    valid_frame = frame[frame.divisao == "validacao"]
    if treino_frame.empty or valid_frame.empty:
        raise ValueError("Manifesto sem divisões treino/validação.")
    treino_loader = _lote(lote, treino_frame, base, True, workers, dev)
    valid_loader = _lote(lote, valid_frame, base, False, workers, dev)
    frequencias = treino_frame.classe.value_counts()
    pesos = torch.tensor([len(treino_frame) / (len(CLASSES) * frequencias.get(c, 1)) for c in CLASSES],
                         dtype=torch.float32, device=dev)
    criterio = nn.CrossEntropyLoss(weight=pesos)
    if taxa is None:
        taxa = 1e-4 if arquitetura == "resnet18" else 1e-3
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=taxa, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    melhor, sem_melhora, historico = -1.0, 0, []
    destino = Path(saida)
    destino.parent.mkdir(parents=True, exist_ok=True)
    for epoca in range(1, epocas + 1):
        mt = _epoca(modelo, treino_loader, dev, criterio, otimizador, scaler)
        mv = _epoca(modelo, valid_loader, dev, criterio)
        historico.append({"epoca": epoca, "treino": mt, "validacao": mv})
        print(f"Época {epoca:02d}/{epocas} | treino perda {mt['perda']:.4f}, f1 {mt['f1_macro']:.3f} "
              f"| validação perda {mv['perda']:.4f}, f1 {mv['f1_macro']:.3f}", flush=True)
        if mv["f1_macro"] > melhor + 1e-5:
            melhor, sem_melhora = mv["f1_macro"], 0
            torch.save({"arquitetura": arquitetura, "classes": list(CLASSES),
                        "modelo": modelo.state_dict(), "epoca": epoca,
                        "f1_validacao": melhor, "tamanho": TAMANHO, "historico": historico}, destino)
            print(f"  Melhor checkpoint salvo: {destino}")
        else:
            sem_melhora += 1
            if sem_melhora >= paciencia:
                print("Early stopping acionado.")
                break
    destino.with_suffix(".json").write_text(json.dumps(historico, indent=2), encoding="utf-8")
    return destino


def carregar_modelo(caminho: str | Path, dispositivo: str = "auto"):
    dev = dispositivo_escolhido(dispositivo)
    # weights_only evita deserializacao arbitraria na leitura de checkpoints.
    checkpoint = torch.load(caminho, map_location="cpu", weights_only=True)
    if tuple(checkpoint["classes"]) != CLASSES:
        raise ValueError("As classes no checkpoint não correspondem ao projeto.")
    modelo = criar_modelo(checkpoint["arquitetura"], pretreinado=False)
    modelo.load_state_dict(checkpoint["modelo"])
    modelo.to(dev).eval()
    return modelo, dev, checkpoint


@torch.inference_mode()
def prever_rgb(rgb: np.ndarray, modelo, dispositivo):
    img = Image.fromarray(np.uint8(rgb[:, :, :3]), "RGB")
    tensor = transformacao(treino=False)(img).unsqueeze(0).to(dispositivo)
    with torch.autocast(device_type=dispositivo.type, dtype=torch.float16, enabled=dispositivo.type == "cuda"):
        logits = modelo(tensor)
    probs = logits.float().softmax(dim=1)[0].cpu().numpy()
    melhor = int(probs.argmax())
    especie, estado = decompor_classe(CLASSES[melhor])
    prob_deteriorada = float(sum(probs[i] for i, c in enumerate(CLASSES) if c.endswith("deteriorada")))
    return {"classe": CLASSES[melhor], "especie": especie, "estado": estado,
            "probabilidade_classe": float(probs[melhor]),
            "probabilidade_deteriorada": prob_deteriorada,
            "probabilidades": {c: float(probs[i]) for i, c in enumerate(CLASSES)}}
