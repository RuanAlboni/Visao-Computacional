"""Deteccao YOLO + classificacao profunda e classica independentes."""
from pathlib import Path
from datetime import datetime,timezone
from uuid import uuid4
import hashlib,time,math
import numpy as np
from .catalogo import ALIMENTOS,REVERSO,decompor,escolher_estado

def hash_arquivo(caminho):
    h=hashlib.sha256()
    with Path(caminho).open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()

class Motor:
    def __init__(self,pesos_detector,pesos_estado=None,device="cpu",usar_classico=False,limiares_classico=None):
        from ultralytics import YOLO
        p=Path(pesos_detector).expanduser().resolve()
        if not p.is_file():raise FileNotFoundError("Detector ausente. Execute scripts/baixar_detector_alimentos.py primeiro.")
        self.detector=YOLO(str(p));self.device=device
        if self.detector.task not in {"detect","segment"}:raise ValueError("Checkpoint nao e detector")
        nomes=self.detector.names
        self.nomes=dict(enumerate(nomes)) if isinstance(nomes,list) else nomes
        self.meta={"detector":str(p),"detector_sha256":hash_arquivo(p)}
        self.classificador=None
        self.usar_classico=bool(usar_classico)
        self.limiares_classico={}
        if self.usar_classico:
            from .classico import configurar_limiares
            self.limiares_classico, origem=configurar_limiares(limiares_classico)
            self.meta.update({"classico": "Visão computacional clássica: cor + textura + regras", "limiares_classico":self.limiares_classico,"origem_limiares_classico":origem})
        if pesos_estado:
            from .profundo import Classificador
            q=Path(pesos_estado).expanduser().resolve()
            if not q.is_file():raise FileNotFoundError("Classificador selecionado nao existe.")
            self.classificador=Classificador(q,device)
            self.meta.update({"classificador":str(q),"classificador_sha256":hash_arquivo(q),"classes_estado":list(self.classificador.classes)})

    def analisar(self,rgb,produtos=None,conf=0.35,limiar_estado=0.60,agnostic_nms=True,iou=0.50):
        a=np.asarray(rgb)
        if a.ndim!=3 or a.shape[2]!=3 or a.dtype!=np.uint8:raise ValueError("Esperado RGB uint8")
        produtos=list(ALIMENTOS) if produtos is None else list(produtos)
        if not produtos or any(p not in ALIMENTOS for p in produtos):raise ValueError("Selecao de alimentos invalida")
        ids=[int(k) for k,v in self.nomes.items() if str(v).lower() in {ALIMENTOS[p] for p in produtos}]
        if not ids:raise ValueError("Detector nao conhece os alimentos selecionados")
        inicio=time.perf_counter();h,w=a.shape[:2]
        saidas=self.detector.predict(source=np.ascontiguousarray(a[...,::-1]),classes=ids,conf=conf,device=self.device,agnostic_nms=agnostic_nms,iou=iou,verbose=False)
        itens=[]
        for saida in saidas:
            if saida.boxes is None:continue
            for b in saida.boxes:
                cid=int(b.cls[0].item());nome=str(self.nomes[cid]).lower()
                if cid not in ids or nome not in REVERSO:continue
                coords=list(map(float,b.xyxy[0].tolist()))
                if not all(math.isfinite(v) for v in coords):continue
                x1,y1,x2,y2=coords
                x1=max(0,min(w,math.floor(x1)));x2=max(0,min(w,math.ceil(x2)))
                y1=max(0,min(h,math.floor(y1)));y2=max(0,min(h,math.ceil(y2)))
                if x2<=x1 or y2<=y1:continue
                produto=REVERSO[nome]
                item={"alimento":produto,"bbox":[x1,y1,x2,y2],"detection_score":float(b.conf[0].item()),
                      "estado":"nao_avaliado","motivo":"Sem classificador de estado para este alimento.","dias_restantes":None}
                recorte=a[y1:y2,x1:x2]
                if getattr(self,"usar_classico",False):
                    from .classico import classificar_recorte
                    item["classico_especie_regras"]=produto
                    item["classico"]=classificar_recorte(recorte,produto,self.limiares_classico)
                if self.classificador:
                    cobertos={decompor(c)[0] for c in self.classificador.classes}
                    if produto in cobertos:
                        probs=self.classificador.prever(recorte)
                        item.update(escolher_estado(probs,produto,limiar_estado));item["probabilidades"]=probs
                        if "classico" in item and "classe_prevista" in item:
                            previsto,_=decompor(item["classe_prevista"])
                            if previsto!=produto:
                                item["classico_aviso"]="YOLO e ResNet discordam sobre a especie; regras classicas usam a especie do YOLO."
                itens.append(item)
        itens.sort(key=lambda i:(i["bbox"][1],i["bbox"][0]))
        for i,item in enumerate(itens,1):item["id_objeto"]=i
        return {"id":uuid4().hex,"criado_em":datetime.now(timezone.utc).isoformat(),
                "imagem_sha256":hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest(),
                "parametros":{"alimentos":produtos,"conf":conf,"limiar_estado":limiar_estado,"agnostic_nms":bool(agnostic_nms),"iou":iou,"usar_classico":bool(getattr(self,"usar_classico",False))},
                "modelos":self.meta,"duracao_s":round(time.perf_counter()-inicio,4),
                "deteccoes":itens,"total_previsto":len(itens),
                "aviso":"Contagem e estado sao previsoes; nao confirmam seguranca alimentar. Prazo nao estimado."}
