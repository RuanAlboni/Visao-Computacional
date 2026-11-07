"""Triagem visual experimental em mascara de candidato; nao avalia seguranca alimentar."""
import numpy as np
from skimage.color import rgb2hsv

def analisar_tomate(rgb, limiar=None, mascara_objeto=None):
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[2] != 3 or a.dtype != np.uint8:
        raise ValueError("Esperado RGB uint8.")
    if mascara_objeto is None:
        raise ValueError("Mascara do objeto obrigatoria; fundo nao pode participar da classificacao.")
    mask = np.asarray(mascara_objeto, dtype=bool)
    if mask.shape != a.shape[:2] or int(mask.sum()) < 50:
        raise ValueError("Mascara invalida ou area insuficiente.")
    pixels = rgb2hsv(a)[mask]
    hh, ss, vv = pixels.T
    # Essas medidas sao descritores, nao probabilidades ou evidencias de mofo.
    vermelho = ((hh < 0.055) | (hh > 0.94)) & (ss > 0.42) & (vv > 0.30)
    escuro = vv < 0.20
    vermelho_fracao, escuro_fracao = float(vermelho.mean()), float(escuro.mean())
    compativel = vermelho_fracao >= 0.65 and escuro_fracao <= 0.10
    return {
        "estado": "sem_alteracoes_visuais_evidentes" if compativel else "inconclusivo",
        "classe": "candidato_sem_alteracoes_evidentes" if compativel else "candidato_inconclusivo",
        "metricas": {"fracao_vermelha": vermelho_fracao, "fracao_escura": escuro_fracao,
                     "pixels_avaliados": int(mask.sum())},
        "motivo": ("Superficie observada predominantemente vermelha; isso nao confirma identidade nem consumo."
                   if compativel else "Evidencia visual insuficiente; nao e possivel concluir deterioracao."),
        "prob_fresco": None, "confianca": None,
    }
