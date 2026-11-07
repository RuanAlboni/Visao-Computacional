"""Caixas numeradas com margem superior, sem encobrir a imagem."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def _fonte(tamanho):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", tamanho)
    except OSError:
        return ImageFont.load_default()

def desenhar(imagem_rgb, resultados, titulo=None, espessura=3):
    original = Image.fromarray(np.asarray(imagem_rgb)).convert("RGB")
    margem = 44
    imagem = Image.new("RGB", (original.width,original.height+margem),(25,25,25))
    imagem.paste(original,(0,margem))
    d = ImageDraw.Draw(imagem)
    fonte = _fonte(max(12,min(24,original.width//45)))
    d.text((6,8),titulo or "Candidatos; identidade e consumo inconclusivos",font=fonte,fill="white")
    for i,r in enumerate(resultados,1):
        x1,y1,x2,y2=r.bbox
        cor=(255,170,0)
        d.rectangle((x1,y1+margem,x2-1,y2-1+margem),outline=cor,width=espessura)
        rotulo=f"#{i} candidato"
        b=d.textbbox((0,0),rotulo,font=fonte)
        tw,th=b[2]-b[0]+10,b[3]-b[1]+10
        x=max(0,min(x1,imagem.width-tw))
        y=max(margem,min(y1+margem,imagem.height-th))
        d.rectangle((x,y,x+tw,y+th),fill=cor)
        d.text((x+4,y+3-b[1]),rotulo,font=fonte,fill="black")
    return imagem

def salvar(imagem_rgb,resultados,destino,titulo=None):
    caminho=Path(destino); caminho.parent.mkdir(parents=True,exist_ok=True)
    desenhar(imagem_rgb,resultados,titulo).save(caminho)
    return caminho
