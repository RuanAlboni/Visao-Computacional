
"""Renderiza uma unica fonte por imagem, vinculada ao ID da analise."""
import numpy as np
from PIL import Image,ImageDraw,ImageFont

def desenhar(rgb,analise,fonte):
    if fonte not in ("candidatos_classicos","deteccoes_do_modelo"):
        raise ValueError("Fonte invalida.")
    original=Image.fromarray(np.asarray(rgb)).convert("RGB")
    im=Image.new("RGB",(original.width,original.height+70),(20,20,20));im.paste(original,(0,70))
    d=ImageDraw.Draw(im)
    try: fonte_texto=ImageFont.truetype("DejaVuSans.ttf",max(12,min(22,original.width//50)))
    except OSError: fonte_texto=ImageFont.load_default()
    d.text((5,5),"ID: "+analise["id"],font=fonte_texto,fill="white")
    status=analise["status_modelo" if fonte=="deteccoes_do_modelo" else "status_classico"]
    descricao=f"{fonte} | {status} | {len(analise[fonte])} regioes"
    d.text((5,34),descricao,font=fonte_texto,fill="white")
    for i,obj in enumerate(analise[fonte],1):
        x1,y1,x2,y2=obj["bbox"]
        cor=(40,180,255) if fonte=="deteccoes_do_modelo" else (255,170,0)
        d.rectangle((x1,y1+70,x2-1,y2+69),outline=cor,width=3)
        texto=f"#{i}"+(f" {obj['confidence']:.2f}" if "confidence" in obj else " candidato")
        b=d.textbbox((0,0),texto,font=fonte_texto);ww=b[2]-b[0]+8;hh=b[3]-b[1]+8
        x=max(0,min(x1,im.width-ww));y=max(70,min(y1+70,im.height-hh))
        d.rectangle((x,y,x+ww,y+hh),fill=cor)
        d.text((x+4,y+3-b[1]),texto,font=fonte_texto,fill="black")
    return im
