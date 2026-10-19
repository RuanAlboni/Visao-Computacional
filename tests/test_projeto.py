"""Testes com imagens sintéticas. Não medem acurácia real do projeto."""
from pathlib import Path
import numpy as np
import cv2
import torch
from src.classico import analisar_classico, sobreposicao
from src.constantes import CLASSES
from src.dados import preparar, carregar_manifesto
from src.profundo import criar_modelo, carregar_modelo, prever_rgb


def fruta_sintetica(especie="maca", deteriorada=False):
    img = np.full((144, 144, 3), 248, dtype=np.uint8)
    if especie == "banana":
        cv2.ellipse(img, (73, 72), (51, 22), -25, 0, 360, (240, 220, 60), -1)
    elif especie == "maca":
        cv2.circle(img, (72, 73), 52, (193, 45, 44), -1)
    else:
        cv2.circle(img, (72, 73), 52, (235, 135, 28), -1)
    if deteriorada:
        cv2.circle(img, (60, 60), 19, (95, 66, 38), -1)
        cv2.circle(img, (92, 84), 13, (67, 45, 23), -1)
    return img


def test_classico_mascara_e_sobreposicao():
    for especie in ("maca", "banana", "laranja"):
        for rotten in (False, True):
            img = fruta_sintetica(especie, rotten)
            res = analisar_classico(img, especie)
            assert res.especie == especie
            assert 0 <= res.fracao_manchas <= 1
            assert 0 <= res.pontuacao <= 1
            assert res.mascara_fruta.sum() > 500
            assert sobreposicao(res).shape == img.shape


def test_arquiteturas_e_checkpoint(tmp_path):
    for nome in ("cnn", "resnet18"):
        model = criar_modelo(nome, pretreinado=False).eval()
        with torch.inference_mode():
            saida = model(torch.randn(1, 3, 224, 224))
        assert saida.shape == (1, 6)
        destino = tmp_path / f"{nome}.pth"
        torch.save({"arquitetura": nome, "classes": list(CLASSES), "modelo": model.state_dict(),
                    "epoca": 1}, destino)
        model2, device, _ = carregar_modelo(destino, "cpu")
        pred = prever_rgb(fruta_sintetica(), model2, device)
        assert pred["classe"] in CLASSES
        assert abs(sum(pred["probabilidades"].values()) - 1) < 1e-5


def test_preparacao_sem_vazamento_por_grupo(tmp_path):
    base = tmp_path / "bruto" / "dataset"
    nomes_pastas = {"maca_fresca": "freshapples", "banana_fresca": "freshbanana",
                     "laranja_fresca": "freshoranges", "maca_deteriorada": "rottenapples",
                     "banana_deteriorada": "rottenbanana", "laranja_deteriorada": "rottenoranges"}
    for pasta in ("train", "test"):
        for classe, nome in nomes_pastas.items():
            caminho = base / pasta / nome
            caminho.mkdir(parents=True)
            especie, estado = classe.split("_")
            for indice in range(5 if pasta == "train" else 2):
                img = fruta_sintetica(especie, estado == "deteriorada")
                cv2.imwrite(str(caminho / f"foto_{indice}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    manifesto = preparar(tmp_path / "bruto", tmp_path / "dados", validacao=0.2)
    assert set(manifesto.divisao.unique()) == {"treino", "validacao", "teste"}
    assert len(manifesto) == (5 + 2) * 6
    assert len(set(manifesto[manifesto.divisao == "treino"].grupo) &
               set(manifesto[manifesto.divisao == "validacao"].grupo)) == 0
    carregado, root = carregar_manifesto(tmp_path / "dados")
    assert len(carregado) == len(manifesto)
    assert root.exists()
