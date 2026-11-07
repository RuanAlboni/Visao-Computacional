"""Saida visual: caixas coloridas pelo veredito e legenda com os dias restantes."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt

CORES = {"apropriado": "#2e7d32", "inapropriado": "#c62828"}


def salvar_anotado(imagem_rgb, resultados, caminho, titulo=None):
    altura, largura = imagem_rgb.shape[:2]
    figura, eixo = plt.subplots(figsize=(largura / 100, altura / 100), dpi=100)
    eixo.imshow(imagem_rgb)
    eixo.axis("off")
    for resultado in resultados:
        if not resultado.bbox:
            continue
        x1, y1, x2, y2 = resultado.bbox
        cor = CORES["apropriado" if resultado.apropriado else "inapropriado"]
        eixo.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         linewidth=3, edgecolor=cor, facecolor="none"))
        rotulo = f"{resultado.veredito} | {resultado.dias_restantes}d | {resultado.confianca:.2f}"
        eixo.text(x1, max(y1 - 8, 12), rotulo, color="white", fontsize=10,
                  bbox=dict(facecolor=cor, edgecolor="none", alpha=0.85, pad=2))
    if titulo:
        eixo.set_title(titulo)
    figura.tight_layout()
    figura.savefig(caminho, bbox_inches="tight")
    plt.close(figura)
    return caminho