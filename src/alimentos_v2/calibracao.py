"""Calibração do método clássico somente em val/ (sem alterar regras HSV/LBP).

O score de cor/textura é fixo; por espécie, selecionamos o ponto de corte
que maximiza F1 macro sobre fotos rotuladas da validação.
"""
from __future__ import annotations

import csv
import cv2
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from .catalogo import decompor
from .conjuntos import entradas
from .classico_core import LIMIARES_PADRAO, abrir_rgb, analisar_classico

EXTENSOES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def melhor_limiar(scores, estados) -> tuple[float, float]:
    """Aplica a regra clássica `score > limiar => deteriorado`."""
    scores = np.asarray(scores, dtype=np.float64)
    estados = np.asarray(estados)
    if len(scores) == 0 or len(scores) != len(estados):
        raise ValueError("Scores e rótulos vazios ou inconsistentes.")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores clássicos precisam estar no intervalo [0, 1].")
    if set(estados) != {"fresco", "deteriorado"}:
        raise ValueError("Calibração requer exemplos frescos e deteriorados da mesma espécie.")
    candidatos = np.unique(np.concatenate((
        [0.0, 1.0],
        np.quantile(scores, np.linspace(0, 1, 301)),
    )))
    melhor = -1.0
    melhores = []
    for limiar in candidatos:
        previstos = np.where(scores > limiar, "deteriorado", "fresco")
        f1 = float(f1_score(estados, previstos, labels=["fresco", "deteriorado"],
                            average="macro", zero_division=0))
        if f1 > melhor + 1e-10:
            melhor, melhores = f1, [float(limiar)]
        elif abs(f1 - melhor) <= 1e-10:
            melhores.append(float(limiar))
    return float(np.median(melhores)), melhor


def calibrar(
    dados: str | Path,
    saida: str | Path = "models/limiares_classico_novos.json",
    limite: int = 0,
    csv_saida: str | Path | None = None,
) -> Path:
    """Lê apenas dados/val/<classe>/*.jpg; nunca utiliza train/ nem test/."""
    dados, saida = Path(dados), Path(saida)
    if limite < 0:
        raise ValueError("--limite precisa ser zero ou positivo.")

    exemplos: dict[str, list[tuple[str, Path]]] = {especie: [] for especie in LIMIARES_PADRAO}
    contagem_por_classe = {}
    for classe, caminho in entradas(dados, "val"):
        especie, estado = decompor(classe)
        if especie not in exemplos:
            continue  # Cenoura e brócolis não possuem regras de deterioração.
        if limite:
            contagem_por_classe[classe] = contagem_por_classe.get(classe, 0) + 1
            if contagem_por_classe[classe] > limite:
                continue
        exemplos[especie].append((estado, caminho))

    # Não sobrescreve calibração existente com resultados parciais.
    for especie, itens in exemplos.items():
        estados = {estado for estado, _ in itens}
        if estados != {"fresco", "deteriorado"}:
            raise ValueError(f"Validação incompleta para {especie}; é necessário ter imagens frescas e deterioradas.")

    coletadas = []
    falhas = []
    for especie, itens in exemplos.items():
        for indice, (estado, caminho) in enumerate(itens, 1):
            try:
                r = analisar_classico(abrir_rgb(caminho), especie)
                if not math.isfinite(r.pontuacao):
                    raise ValueError("Pontuação inválida (NaN/Inf).")
                coletadas.append({"arquivo": str(caminho), "especie": especie,
                                  "estado_real": estado, "pontuacao": r.pontuacao})
            except (ValueError, OSError, RuntimeError, cv2.error) as e:
                falhas.append({"arquivo": str(caminho), "erro": str(e)})
            if indice % 200 == 0:
                print(f"Calibração {especie}: {indice}/{len(itens)} imagens", flush=True)

    limites, f1s, contagens = {}, {}, {}
    for especie in LIMIARES_PADRAO:
        grupo = [r for r in coletadas if r["especie"] == especie]
        if {r["estado_real"] for r in grupo} != {"fresco", "deteriorado"}:
            raise ValueError(f"Imagens válidas insuficientes para calibrar {especie}; arquivo não foi alterado.")
        limite_optimo, f1 = melhor_limiar(
            [r["pontuacao"] for r in grupo], [r["estado_real"] for r in grupo]
        )
        limites[especie] = limite_optimo
        f1s[especie] = f1
        contagens[especie] = len(grupo)
        print(f"{especie}: limiar={limite_optimo:.6f} | F1 macro validação={f1:.4f} | n={len(grupo)}")

    resultado = {
        "limiares": limites,
        "f1_validacao": f1s,
        "amostras": len(coletadas),
        "amostras_por_especie": contagens,
        "falhas_segmentacao": len(falhas),
        "metodo": "Limiares maximizando F1-macro apenas na validação; score=0.85*manchas+0.15*textura.",
    }
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    if csv_saida is not None:
        csv_saida = Path(csv_saida)
        csv_saida.parent.mkdir(parents=True, exist_ok=True)
        with csv_saida.open("w", newline="", encoding="utf-8") as f:
            campos = ["arquivo", "especie", "estado_real", "pontuacao"]
            writer = csv.DictWriter(f, fieldnames=campos)
            writer.writeheader()
            writer.writerows(coletadas)
    print(f"Calibração salva em: {saida} (falhas de segmentação: {len(falhas)})")
    return saida
