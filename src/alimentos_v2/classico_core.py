"""Classificador interpretable: segmentacao + cor + textura + regra.

Nao usa redes neurais, SVM nem modelos aprendidos. A calibracao escolhe apenas
um limiar escalar por especie a partir do conjunto de validacao.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageOps
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

LIMIARES_PADRAO = {"maca": 0.055, "banana": 0.095, "laranja": 0.060}

@dataclass
class ResultadoClassico:
    especie: str
    estado: str
    pontuacao: float
    limiar: float
    fracao_manchas: float
    fracao_textura: float
    lbp_entropia: float
    glcm_contraste: float
    glcm_homogeneidade: float
    imagem: np.ndarray
    mascara_fruta: np.ndarray
    mascara_suspeita: np.ndarray
    mascara_textura: np.ndarray
    aviso: str = ""


def abrir_rgb(caminho: str | Path) -> np.ndarray:
    with Image.open(caminho) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))


def reduzir(rgb: np.ndarray, limite: int = 384) -> np.ndarray:
    altura, largura = rgb.shape[:2]
    fator = min(1.0, limite / max(altura, largura))
    if fator >= 1:
        return rgb.copy()
    return cv2.resize(rgb, (max(1, int(largura * fator)), max(1, int(altura * fator))), interpolation=cv2.INTER_AREA)


def _componente_principal(mascara: np.ndarray) -> np.ndarray:
    mascara = np.asarray(mascara, dtype=np.uint8)
    n, rotulos, estatisticas, _ = cv2.connectedComponentsWithStats(mascara, 8)
    if n <= 1:
        return np.zeros(mascara.shape, dtype=np.uint8)
    indice = 1 + np.argmax(estatisticas[1:, cv2.CC_STAT_AREA])
    saida = (rotulos == indice).astype(np.uint8) * 255
    # Fecha buracos internos da fruta.
    contornos, _ = cv2.findContours(saida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contornos:
        cv2.drawContours(saida, contornos, -1, 255, cv2.FILLED)
    return saida


def segmentar_fruta(rgb: np.ndarray) -> tuple[np.ndarray, str]:
    """Segmentacao por diferenca do fundo nos cantos; GrabCut como reserva."""
    h, w = rgb.shape[:2]
    if min(h, w) < 35:
        raise ValueError("A imagem precisa ter ao menos 35 pixels em cada dimensao.")
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
        return principal > 0, "Segmentação por diferença de cor em relação ao fundo."

    # GrabCut supoe fruta mais centralizada e alguma borda de fundo.
    margem = max(2, int(min(h, w) * 0.055))
    retangulo = (margem, margem, w - 2 * margem, h - 2 * margem)
    try:
        imagem_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        mascara_gc = np.zeros((h, w), dtype=np.uint8)
        bg = np.zeros((1, 65), np.float64)
        fg = np.zeros((1, 65), np.float64)
        cv2.grabCut(imagem_bgr, mascara_gc, retangulo, bg, fg, 4, cv2.GC_INIT_WITH_RECT)
        m = np.where((mascara_gc == cv2.GC_FGD) | (mascara_gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
        principal = _componente_principal(m)
        fracao = np.count_nonzero(principal) / (h * w)
        if 0.035 <= fracao <= 0.95:
            return principal > 0, "Segmentação com GrabCut (fundo difícil)."
    except cv2.error:
        pass
    raise ValueError("Não foi possível isolar a fruta. Use uma foto de uma fruta centralizada com fundo relativamente uniforme.")


def _textura(gray: np.ndarray, mascara_fruta: np.ndarray, regiao_escura: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    # Residuo de alta frequencia: imperfeicoes com contraste local alto.
    borrada = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.4)
    residuo = cv2.absdiff(gray, borrada)
    ruido = (residuo > 14) & regiao_escura & mascara_fruta
    # LBP/GLCM sao medidas descritivas na interface (nao "probabilidades").
    gray_pequena = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    mask_pequena = cv2.resize(mascara_fruta.astype(np.uint8), (96, 96), interpolation=cv2.INTER_NEAREST) > 0
    lbp = local_binary_pattern(gray_pequena, P=8, R=1, method="uniform")
    valores = lbp[mask_pequena]
    hist = np.bincount(valores.astype(np.int32), minlength=10).astype(np.float64)
    hist = hist / max(1.0, hist.sum())
    entropia = float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))
    # Fundo neutro para nao influenciar tanto a coocorrencia.
    nivel = (gray_pequena // 8).astype(np.uint8)
    if np.any(mask_pequena):
        nivel[~mask_pequena] = int(np.median(nivel[mask_pequena]))
    glcm = graycomatrix(nivel, distances=[1], angles=[0, np.pi / 2], levels=32, symmetric=True, normed=True)
    contraste = float(np.mean(graycoprops(glcm, "contrast")))
    homogeneidade = float(np.mean(graycoprops(glcm, "homogeneity")))
    return ruido, entropia, contraste, homogeneidade


def analisar_classico(rgb: np.ndarray, especie: str, limiar: float | None = None) -> ResultadoClassico:
    if especie not in LIMIARES_PADRAO:
        raise ValueError(f"Espécie não suportada: {especie}")
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
        raise ValueError("É esperada uma imagem RGB.")
    rgb = reduzir(np.asarray(rgb[:, :, :3], dtype=np.uint8))
    mascara, info = segmentar_fruta(rgb)
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

    if especie == "banana":
        suspeita = preto | (marrom & escuro_relativo) | ((v < 115) & escuro_relativo)
    elif especie == "maca":
        suspeita = preto | (marrom & escuro_relativo & (v < 160)) | (cinza_mofo & (v < 135))
    else:
        verde_mofo = (h >= 34) & (h <= 95) & (s > 48) & (v < 210)
        suspeita = preto | verde_mofo | (marrom & escuro_relativo & (v < 145)) | (cinza_mofo & (v < 130))

    suspeita = (suspeita & mascara).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    suspeita = cv2.morphologyEx(suspeita, cv2.MORPH_OPEN, k)
    # Ignora regioes minimas e espurias (< 0,08% da fruta).
    n, rot, stats, _ = cv2.connectedComponentsWithStats(suspeita, 8)
    minimo = max(4, round(np.count_nonzero(mascara) * 0.0008))
    limpa = np.zeros_like(suspeita)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= minimo:
            limpa[rot == i] = 255
    manchas = limpa > 0
    regiao_escura = escuro_relativo | (v < 110) | manchas
    rugosidade, entropia, contraste, homogeneidade = _textura(gray, mascara, regiao_escura)
    total = int(np.count_nonzero(mascara))
    fracao_manchas = float(np.count_nonzero(manchas)) / max(1, total)
    fracao_textura = float(np.count_nonzero(rugosidade)) / max(1, total)
    pontuacao = float(np.clip(0.85 * fracao_manchas + 0.15 * fracao_textura, 0, 1))
    t = LIMIARES_PADRAO[especie] if limiar is None else float(limiar)
    return ResultadoClassico(
        especie=especie, estado="deteriorada" if pontuacao > t else "fresca",
        pontuacao=pontuacao, limiar=t, fracao_manchas=fracao_manchas,
        fracao_textura=fracao_textura, lbp_entropia=entropia,
        glcm_contraste=contraste, glcm_homogeneidade=homogeneidade,
        imagem=rgb, mascara_fruta=mascara, mascara_suspeita=manchas,
        mascara_textura=rugosidade, aviso=info,
    )


def sobreposicao(resultado: ResultadoClassico) -> np.ndarray:
    img = resultado.imagem.copy()
    sombra = (img.astype(np.float32) * 0.55).astype(np.uint8)
    img[~resultado.mascara_fruta] = sombra[~resultado.mascara_fruta]
    # Vermelho: possiveis manchas; amarelo: textura irregular.
    somente_textura = resultado.mascara_textura & ~resultado.mascara_suspeita
    img[somente_textura] = (0.57 * img[somente_textura] + 0.43 * np.array([245, 189, 45])).astype(np.uint8)
    img[resultado.mascara_suspeita] = (0.43 * img[resultado.mascara_suspeita] + 0.57 * np.array([248, 62, 65])).astype(np.uint8)
    return img


def carregar_limiares(caminho: str | Path) -> dict[str, float]:
    caminho = Path(caminho)
    if not caminho.is_file():
        return LIMIARES_PADRAO.copy()
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    valores = dados.get("limiares", dados)
    return {k: float(valores.get(k, LIMIARES_PADRAO[k])) for k in LIMIARES_PADRAO}
