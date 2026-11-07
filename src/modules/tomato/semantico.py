
"""Adaptador YOLO de deteccao. Nao avalia estado ou consumo."""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import math
import unicodedata
import numpy as np

class DetectorIncompativel(ValueError):
    pass

def normalizar(nome):
    return unicodedata.normalize("NFKD",str(nome)).encode("ascii","ignore").decode().strip().casefold()

@dataclass
class Deteccao:
    bbox: tuple
    classe_id: int
    classe_nome: str
    confidence: float
    fonte: str = "modelo"

    def para_dict(self):
        return asdict(self)

def sha256_arquivo(caminho):
    h=hashlib.sha256()
    with Path(caminho).open("rb") as f:
        for bloco in iter(lambda:f.read(1024*1024),b""): h.update(bloco)
    return h.hexdigest()

class DetectorTomate:
    def __init__(self, modelo, nomes_aceitos=("tomato","tomatoes","tomate","tomates")):
        if getattr(modelo,"task",None) not in ("detect","segment"):
            raise DetectorIncompativel("Checkpoint nao e de deteccao/segmentacao de objetos.")
        nomes=getattr(modelo,"names",None)
        if isinstance(nomes,(list,tuple)): nomes=dict(enumerate(nomes))
        if not isinstance(nomes,dict): raise DetectorIncompativel("Mapa de classes ausente.")
        self.nomes={int(k):str(v) for k,v in nomes.items()}
        aceitos={normalizar(v) for v in nomes_aceitos}
        self.ids=[k for k,v in self.nomes.items() if normalizar(v) in aceitos]
        if not self.ids:
            raise DetectorIncompativel("Checkpoint sem classe explicita de tomate. Classes: "+str(self.nomes))
        self.modelo=modelo

    @classmethod
    def carregar(cls,caminho,device="cpu"):
        p=Path(caminho).expanduser().resolve()
        if not p.is_file(): raise FileNotFoundError(f"Pesos locais ausentes: {p}")
        if p.suffix.lower()!=".pt": raise DetectorIncompativel("Nesta revisao use checkpoint local .pt confiavel.")
        from ultralytics import YOLO
        objeto=cls(YOLO(str(p)))
        objeto.device=device
        objeto.caminho=str(p)
        objeto.sha256=sha256_arquivo(p)
        return objeto

    def prever(self,rgb,conf=0.35,iou=0.5):
        a=np.asarray(rgb)
        if a.ndim!=3 or a.shape[2]!=3 or a.dtype!=np.uint8:
            raise ValueError("Esperado RGB uint8 HxWx3.")
        if not (0<conf<=1 and 0<iou<=1): raise ValueError("conf e iou devem estar em (0,1].")
        h,w=a.shape[:2]
        # Ultralytics recebe ndarray em BGR, nao RGB.
        resultados=self.modelo.predict(source=np.ascontiguousarray(a[...,::-1]),
            classes=self.ids,conf=conf,iou=iou,device=getattr(self,"device","cpu"),verbose=False)
        deteccoes=[]
        for resultado in resultados:
            caixas=getattr(resultado,"boxes",None)
            if caixas is None: continue
            for caixa in caixas:
                cid=int(caixa.cls[0].item())
                score=float(caixa.conf[0].item())
                coords=[float(v) for v in caixa.xyxy[0].tolist()]
                if cid not in self.ids or not math.isfinite(score) or score<conf: continue
                if len(coords)!=4 or not all(math.isfinite(v) for v in coords): continue
                x1,y1,x2,y2=coords
                x1=max(0,min(w,math.floor(x1)));x2=max(0,min(w,math.ceil(x2)))
                y1=max(0,min(h,math.floor(y1)));y2=max(0,min(h,math.ceil(y2)))
                if x2<=x1 or y2<=y1: continue
                deteccoes.append(Deteccao((x1,y1,x2,y2),cid,self.nomes[cid],score))
        deteccoes.sort(key=lambda d:(d.bbox[1],d.bbox[0]))
        return deteccoes
