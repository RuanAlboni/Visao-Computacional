"""Integra as regras classicas de cor e textura a recortes do YOLO.

Nao utiliza rede neural. A configuracao de especie e independente da ResNet:
por padrao, as regras escolhidas correspondem ao rotulo do detector YOLO.
Nenhuma informacao visualizavel (arrays) e colocada no relatorio JSON.
"""
from pathlib import Path
import math
import cv2

from .classico_core import (
    LIMIARES_PADRAO,
    analisar_classico,
    carregar_limiares,
    sobreposicao,
)

FRUTAS_SUPORTADAS = frozenset(LIMIARES_PADRAO)


def configurar_limiares(caminho=None):
    """Retorna (limiares, origem). Sem JSON, usa heuristicas de referencia."""
    if caminho:
        arquivo = Path(caminho).expanduser().resolve()
        if arquivo.is_file():
            valores = carregar_limiares(arquivo)
            origem = str(arquivo)
        else:
            valores = LIMIARES_PADRAO.copy()
            origem = "padrao (arquivo de calibracao ausente)"
    else:
        valores = LIMIARES_PADRAO.copy()
        origem = "padrao (sem arquivo de calibracao)"
    for especie, valor in valores.items():
        if especie not in FRUTAS_SUPORTADAS or not math.isfinite(valor) or not 0 <= valor <= 1:
            raise ValueError(f"Limiar classico invalido para {especie}: {valor}")
    return valores, origem


def classificar_recorte(rgb, especie, limiares):
    """Resultado serializavel: nunca confundir score heuristico com probabilidade."""
    if especie not in FRUTAS_SUPORTADAS:
        return {"estado": "nao_avaliado", "motivo": "Sem regras classicas para este alimento."}
    try:
        r = analisar_classico(rgb, especie, limiares[especie])
    except (ValueError, RuntimeError, cv2.error) as e:
        return {"estado": "inconclusivo", "motivo": f"Falha de segmentacao/análise: {e}"}
    return {
        "estado": "deteriorado" if r.estado == "deteriorada" else "fresco",
        "pontuacao": round(r.pontuacao, 6),
        "limiar": r.limiar,
        "fracao_manchas": round(r.fracao_manchas, 6),
        "fracao_textura": round(r.fracao_textura, 6),
        "lbp_entropia": round(r.lbp_entropia, 6),
        "glcm_contraste": round(r.glcm_contraste, 6),
        "glcm_homogeneidade": round(r.glcm_homogeneidade, 6),
        "segmentacao": r.aviso,
        "motivo": "Regras de cor/textura; pontuacao NAO e probabilidade.",
    }


def visualizar_recorte(rgb, especie, limiar):
    """Retorna (imagem RGB com manchas, mascara da fruta, mascara suspeita)."""
    if especie not in FRUTAS_SUPORTADAS:
        raise ValueError("Sem metodo classico para este alimento")
    r = analisar_classico(rgb, especie, limiar)
    return sobreposicao(r), r.mascara_fruta, r.mascara_suspeita
