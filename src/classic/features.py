"""Extração de features clássicas (HSV, LBP, GLCM, manchas).
Adaptado de FruitVision/src/classico.py — apenas funções de extração,
sem lógica de decisão."""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def abrir_rgb(caminho):
    from PIL import Image, ImageOps
    with Image.open(caminho) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))

def reduzir(rgb, limite=384):
    altura, largura = rgb.shape[:2]
    fator = min(1.0, limite / max(altura, largura))
    if fator >= 1:
        return rgb.copy()
    return cv2.resize(rgb, (max(1, int(largura * fator)), max(1, int(altura * fator))), interpolation=cv2.INTER_AREA)

def segmentar_fruta(rgb):
    """Segmentação por diferença do fundo; GrabCut como reserva."""
    h, w = rgb.shape[:2]
    if min(h, w) < 35:
        raise ValueError("Imagem muito pequena.")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    b = max(3, min(h, w) // 12)
    cantos = np.concatenate([
        lab[:b, :b].reshape(-1, 3), lab[:b, -b:].reshape(-1, 3),
        lab[-b:, :b].reshape(-1, 3), lab[-b:, -b:].reshape(-1, 3)
    ], axis=0)
    cor_fundo = np.median(cantos.astype(np.float32), axis=0)
    distancia = np.linalg.norm(lab.astype(np.float32) - cor_fundo, axis=2)
    saturacao = hsv[:, :, 1]
    brilho = hsv[:, :, 2]
    candidato = ((distancia > 29) | ((saturacao > 65) & (brilho > 38) & (distancia > 15))).astype(np.uint8) * 255
    tamanho = max(3, int(min(h, w) * 0.017) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamanho, tamanho))
    candidato = cv2.morphologyEx(candidato, cv2.MORPH_OPEN, kernel)
    candidato = cv2.morphologyEx(candidato, cv2.MORPH_CLOSE, kernel, iterations=2)
    principal = _componente_principal(candidato)
    fracao = np.count_nonzero(principal) / (h * w)
    if 0.035 <= fracao <= 0.88:
        return principal > 0, "Segmentação por diferença de fundo."
    raise ValueError("Não foi possível isolar a fruta.")

def _componente_principal(mascara):
    mascara = np.asarray(mascara, dtype=np.uint8)
    n, rotulos, estatisticas, _ = cv2.connectedComponentsWithStats(mascara, 8)
    if n <= 1:
        return np.zeros(mascara.shape, dtype=np.uint8)
    indice = 1 + np.argmax(estatisticas[1:, cv2.CC_STAT_AREA])
    saida = (rotulos == indice).astype(np.uint8) * 255
    contornos, _ = cv2.findContours(saida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        cv2.drawContours(saida, contornos, -1, 255, cv2.FILLED)
    return saida

def extrair_features(rgb, mascara_fruta):
    """Extrai features: manchas, textura, LBP, GLCM."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    h, s, v = (hsv[:, :, i].astype(np.int16) for i in range(3))
    l = lab[:, :, 0].astype(np.int16)
    ref_claro = np.percentile(l[mascara], 78)
    escuro_relativo = (ref_claro - l) > 24
    preto = v < 82
    marrom = (h >= 7) & (h <= 27) & (s > 50) & (v < 177)
    cinza_mofo = (s < 66) & (v < 153) & escuro_relativo
    return {"preto": preto, "marrom": marrom, "cinza_mofo": cinza_mofo,
            "escuro_relativo": escuro_relativo, "gray": gray, "v": v}
