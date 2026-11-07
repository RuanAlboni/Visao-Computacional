"""Regras de decisão do classificador clássico.
Adaptado de FruitVision — parametrizado por fruta."""
import json
from pathlib import Path
import cv2
import numpy as np
from src.classic.features import extrair_features

LIMIARES_PADRAO = {"tomato": 0.055, "apple": 0.055, "strawberry": 0.060}

def classificar_classico(rgb, mascara_fruta, fruit_type="apple", limiar=None):
    """Classifica fruta como fresh/rotten baseado em regras."""
    feats = extrair_features(rgb, mascara_fruta)
    # Regras específicas por fruta (adaptar conforme necessário)
    suspeita = feats["preto"] | (feats["marrom"] & feats["escuro_relativo"]) | (feats["cinza_mofo"] & (feats["v"] < 135))
    suspeita = (suspeita & mascara_fruta).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    suspeita = cv2.morphologyEx(suspeita, cv2.MORPH_OPEN, k)
    total = int(np.count_nonzero(mascara))
    fracao_manchas = float(np.count_nonzero(suspeita > 0)) / max(1, total)
    pontuacao = float(np.clip(0.85 * fracao_manchas, 0, 1))
    t = LIMIARES_PADRAO.get(fruit_type, 0.055) if limiar is None else float(limiar)
    estado = "rotten" if pontuacao > t else "fresh"
    return {"condition": estado, "score": pontuacao, "threshold": t,
            "spot_fraction": fracao_manchas}
