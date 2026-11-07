"""Testes dos fluxos novos sem download de dataset nem pré-treino."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.alimentos_v2 import calibracao, conjuntos, treinamento
from src.alimentos_v2.profundo import Classificador

CLASSES = [f"{p}_{e}" for p in ("maca", "banana", "laranja") for e in ("fresca", "deteriorada")]


def criar_imagens(raiz):
    for divisao in ("train", "val", "test"):
        for i, classe in enumerate(CLASSES):
            pasta = raiz / divisao / classe
            pasta.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.full((40, 40, 3), (i * 32) % 255, np.uint8)).save(pasta / f"{i}.png")


def test_melhor_limiar_f1():
    t, f1 = calibracao.melhor_limiar([.01, .02, .2, .3],
                                   ["fresco", "fresco", "deteriorado", "deteriorado"])
    assert .02 <= t < .2 and f1 == 1.0
    with pytest.raises(ValueError):
        calibracao.melhor_limiar([.01, .02], ["fresco", "fresco"])


def test_calibrar_apenas_validacao(tmp_path, monkeypatch):
    criar_imagens(tmp_path)
    # Pontuação inteiramente controlada para verificar o cálculo dos limiares.
    monkeypatch.setattr(calibracao, "analisar_classico", lambda img, especie: SimpleNamespace(
        pontuacao=0.02 if img[0, 0, 0] % 64 == 32 else 0.2))
    # Testa ainda que `test` e `train` não influenciam: a função não as abre.
    seen = []
    def abrir(path):
        seen.append(str(path))
        classe = path.parent.name
        return np.full((2, 2, 3), 32 if classe.endswith("fresca") else 64, dtype=np.uint8)
    monkeypatch.setattr(calibracao, "abrir_rgb", abrir)
    arquivo = tmp_path / "limiares.json"
    destino = calibracao.calibrar(tmp_path, arquivo, csv_saida=tmp_path / "scores.csv")
    dados = json.loads(destino.read_text(encoding="utf-8"))
    assert len(seen) == 6
    assert all("/val/" in p.replace("\\", "/") for p in seen)
    assert set(dados["limiares"]) == {"maca", "banana", "laranja"}
    assert dados["amostras"] == 6
    assert dados["falhas_segmentacao"] == 0
    assert (tmp_path / "scores.csv").is_file()


def test_manifesto_csv(tmp_path):
    origem = tmp_path / "original"
    origem.mkdir()
    for nome in ("a.png", "b.png", "c.png"):
        Image.fromarray(np.zeros((40, 40, 3), dtype=np.uint8)).save(origem / nome)
    raiz = tmp_path / "dados"
    raiz.mkdir()
    (raiz / "metadados.json").write_text(json.dumps({"origem": str(origem)}), encoding="utf-8")
    with (raiz / "manifesto.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["arquivo", "classe", "divisao"])
        writer.writeheader()
        for nome, divisao in [("a.png", "treino"), ("b.png", "validacao"), ("c.png", "teste")]:
            writer.writerow({"arquivo": nome, "classe": "maca_fresca", "divisao": divisao})
    assert [p.name for _, p in conjuntos.entradas(raiz, "train")] == ["a.png"]
    assert [p.name for _, p in conjuntos.entradas(raiz, "val")] == ["b.png"]
    assert [p.name for _, p in conjuntos.entradas(raiz, "test")] == ["c.png"]


def test_treino_resnet_salva_checkpoint_compativel(tmp_path, monkeypatch):
    criar_imagens(tmp_path)
    # Imitamos duas épocas rápidas para testar early stopping + formato sem custo de backprop.
    metricas = iter([
        {"perda": 1.2, "acuracia": .55, "f1_macro": .5},
        {"perda": .9, "acuracia": .80, "f1_macro": .8},
        {"perda": 1.1, "acuracia": .6, "f1_macro": .6},
        {"perda": 1.0, "acuracia": .7, "f1_macro": .7},
    ])
    monkeypatch.setattr(treinamento, "executar_epoca", lambda *a, **k: next(metricas))
    pesos = tmp_path / "novo.pth"
    treino = treinamento.treinar(tmp_path, pesos, "resnet18", epocas=4, lote=2,
                                paciencia=1, sem_pretreino=True, dispositivo="cpu")
    assert treino.is_file()
    import torch
    ck = torch.load(treino, map_location="cpu", weights_only=True)
    assert ck["epoca"] == 1 and ck["f1_validacao"] == .8
    assert ck["classes"] == sorted(CLASSES)
    hist = json.loads(treino.with_suffix(".historico.json").read_text(encoding="utf-8"))
    assert len(hist) == 2
    classificador = Classificador(treino, "cpu")
    resultado = classificador.prever(np.full((50, 50, 3), 150, dtype=np.uint8))
    assert set(resultado) == set(CLASSES)
    assert sum(resultado.values()) == pytest.approx(1.0, abs=1e-5)
