"""Leitura de dados em pastas train/val OU manifesto CSV com metadados.

Não cria divisões automaticamente para evitar vazamento acidental entre
fotografias originais, augmentations, validação e teste.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image, ImageOps

from .catalogo import decompor, validar_classes

EXTENSOES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _linhas_manifesto(raiz: Path):
    meta = raiz / "metadados.json"
    manifesto = raiz / "manifesto.csv"
    if not meta.is_file():
        raise FileNotFoundError(f"Metadados ausentes: {meta}")
    origem = Path(json.loads(meta.read_text(encoding="utf-8"))["origem"])
    if not origem.is_dir():
        raise FileNotFoundError(f"Pasta original do dataset não existe: {origem}")
    with manifesto.open(encoding="utf-8-sig", newline="") as f:
        linhas = list(csv.DictReader(f))
    linhas_validas = []
    for r in linhas:
        divisao = {"treino": "train", "validacao": "val", "teste": "test"}.get(r.get("divisao"))
        if divisao is None:
            continue
        classe = r["classe"]
        decompor(classe)
        caminho = origem / r["arquivo"]
        if not caminho.is_file():
            raise FileNotFoundError(f"Imagem do manifesto não encontrada: {caminho}")
        linhas_validas.append((divisao, classe, caminho))
    return linhas_validas


def entradas(dados: str | Path, divisao: str):
    """Retorna (classe, caminho) usando somente a divisão solicitada."""
    dados = Path(dados)
    if (dados / "manifesto.csv").is_file():
        return [(classe, path) for d, classe, path in _linhas_manifesto(dados) if d == divisao]
    pasta = dados / divisao
    if not pasta.is_dir():
        raise FileNotFoundError(f"Pasta não encontrada: {pasta} (ou forneça o manifesto.csv original)")
    linhas = []
    for classe_dir in sorted(pasta.iterdir()):
        if not classe_dir.is_dir():
            continue
        try:
            decompor(classe_dir.name)
        except ValueError:
            raise ValueError(f"Nome de pasta de classe não reconhecido: {classe_dir.name}")
        linhas += [(classe_dir.name, p) for p in sorted(classe_dir.rglob("*"))
                   if p.is_file() and p.suffix.lower() in EXTENSOES]
    return linhas


def datasets_treino_validacao(dados, transformacao_treino, transformacao_val):
    from torch.utils.data import Dataset

    treino = entradas(dados, "train")
    validacao = entradas(dados, "val")
    classes = validar_classes(sorted({classe for classe, _ in treino + validacao}))
    mapa = {c: i for i, c in enumerate(classes)}

    class Imagens(Dataset):
        def __init__(self, linhas, transform):
            self.classes = classes
            self.class_to_idx = mapa
            self.samples = [(str(arquivo), mapa[classe]) for classe, arquivo in linhas]
            self.transform = transform

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, indice):
            caminho, label = self.samples[indice]
            with Image.open(caminho) as img:
                rgb = ImageOps.exif_transpose(img).convert("RGB")
            return self.transform(rgb), label

    return Imagens(treino, transformacao_treino), Imagens(validacao, transformacao_val)
