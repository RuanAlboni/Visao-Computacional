"""Propostas de objetos avermelhados; nao confirma identidade de tomate."""
from dataclasses import dataclass
import numpy as np
from scipy import ndimage as ndi
from skimage.color import rgb2hsv
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

@dataclass
class Candidato:
    bbox: tuple
    mascara: np.ndarray
    metodo: str = "cor_watershed"

def validar_rgb(rgb):
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError("Esperado RGB uint8 com formato (altura, largura, 3).")
    if min(a.shape[:2]) < 20:
        raise ValueError("Imagem muito pequena (minimo 20 pixels por lado).")
    return a

def propor_candidatos(rgb, area_min=120, max_candidatos=50):
    a = validar_rgb(rgb)
    h, w = a.shape[:2]
    hsv = rgb2hsv(a)
    vermelho = ((hsv[..., 0] < 0.055) | (hsv[..., 0] > 0.94))
    sementes = vermelho & (hsv[..., 1] > 0.42) & (hsv[..., 2] > 0.20)
    sementes = ndi.binary_opening(sementes, iterations=1)
    sementes = ndi.binary_closing(sementes, iterations=2)
    componentes, n = ndi.label(sementes)
    candidatos, logs = [], []
    minimo = max(int(area_min), int(h*w*0.0003))
    for i in range(1, n+1):
        componente = componentes == i
        if componente.sum() < minimo:
            continue
        yy, xx = np.where(componente)
        x0, x3 = max(0, int(xx.min())-3), min(w, int(xx.max())+4)
        y0, y3 = max(0, int(yy.min())-3), min(h, int(yy.max())+4)
        local = ndi.binary_fill_holes(componente[y0:y3, x0:x3])
        distancia = ndi.distance_transform_edt(local)
        raio = float(distancia.max())
        picos = peak_local_max(distancia, min_distance=max(6, int(raio*0.9)),
                               threshold_abs=max(2, raio*0.55), exclude_border=False)
        marcadores = np.zeros(local.shape, dtype=np.int32)
        for j, (py, px) in enumerate(picos, 1):
            marcadores[py, px] = j
        if not len(picos):
            continue
        rotulos = watershed(-distancia, marcadores, mask=local)
        for j in range(1, len(picos)+1):
            mask = rotulos == j
            area = int(mask.sum())
            if area < minimo:
                continue
            ys, xs = np.where(mask)
            xa, xb = int(xs.min()), int(xs.max())+1
            ya, yb = int(ys.min()), int(ys.max())+1
            largura, altura = xb-xa, yb-ya
            if max(largura/altura, altura/largura) > 2.8:
                logs.append("Regiao alongada rejeitada; pode nao ser tomate.")
                continue
            if area/(h*w) > 0.88:
                logs.append("Regiao ocupa quase toda a imagem; identificacao inconclusiva.")
                continue
            candidatos.append(Candidato((x0+xa, y0+ya, x0+xb, y0+yb), mask[ya:yb, xa:xb]))
    candidatos.sort(key=lambda c:(c.bbox[1], c.bbox[0]))
    if len(candidatos) > max_candidatos:
        logs.append(f"Limite de {max_candidatos} candidatos atingido; contagem parcial.")
        candidatos = candidatos[:max_candidatos]
    logs.append(f"{len(candidatos)} candidato(s) por cor/forma; identidade de tomate nao confirmada.")
    logs.append("Limitacoes: pimentoes vermelhos podem entrar; tomates verdes ou totalmente escuros podem nao aparecer.")
    return candidatos, logs

def detectar_regioes(imagem_rgb, modelo_yolo=None, min_area=120):
    if modelo_yolo is not None:
        raise NotImplementedError("Detector profundo sera integrado com pesos e classes validados no bloco 2.")
    candidatos, _ = propor_candidatos(imagem_rgb, area_min=min_area)
    return [c.bbox for c in candidatos], "candidatos_classicos"
