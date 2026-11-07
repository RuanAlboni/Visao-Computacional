
"""Comparacao de localizacao: candidatos classicos separados das predicoes."""
from datetime import datetime, timezone
from uuid import uuid4
import hashlib
import time
import numpy as np

def executar(rgb,detector=None,executar_classico=True,conf=0.35,iou=0.5,metadados_modelo=None):
    a=np.ascontiguousarray(rgb)
    if a.ndim!=3 or a.shape[2]!=3 or a.dtype!=np.uint8:
        raise ValueError("Imagem RGB uint8 obrigatoria.")
    inicio=time.perf_counter()
    resultado={"id":uuid4().hex,"criado_em":datetime.now(timezone.utc).isoformat(),
        "imagem_sha256":hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest(),
        "dimensoes":{"altura":a.shape[0],"largura":a.shape[1]},
        "parametros":{"conf":conf,"iou":iou,"executar_classico":executar_classico},
        "modelo":metadados_modelo or {},"candidatos_classicos":[],"deteccoes_do_modelo":[],
        "status_classico":"nao_executado","status_modelo":"indisponivel",
        "erros":{},"logs":[],"estado":"nao_avaliado","dias_restantes":None}
    if executar_classico:
        try:
            from src.modules.tomato.deteccao import propor_candidatos
            candidatos,logs=propor_candidatos(a)
            resultado["candidatos_classicos"]=[{"bbox":list(c.bbox),"fonte":"classico"} for c in candidatos]
            resultado["status_classico"]="executado"
            resultado["logs"].extend(logs)
        except Exception as e:
            resultado["status_classico"]="erro";resultado["erros"]["classico"]=str(e)
    if detector is not None:
        try:
            resultado["deteccoes_do_modelo"]=[d.para_dict() for d in detector.prever(a,conf=conf,iou=iou)]
            resultado["status_modelo"]="executado"
        except Exception as e:
            resultado["status_modelo"]="erro";resultado["erros"]["modelo"]=str(e)
    resultado["duracao_s"]=round(time.perf_counter()-inicio,4)
    resultado["logs"].insert(0,f"analise_id={resultado['id']} imagem_sha256={resultado['imagem_sha256']}")
    for chave,status_chave in (("candidatos_classicos","status_classico"),("deteccoes_do_modelo","status_modelo")):
        resultado["logs"].append(f"{chave}: status={resultado[status_chave]} quantidade={len(resultado[chave])}")
        for i,d in enumerate(resultado[chave],1):
            resultado["logs"].append(f"{chave} #{i} bbox={d['bbox']} score={d.get('confidence','nao_estimado')}")
    resultado["logs"].append("Predicoes nao sao anotacoes verdadeiras. Estado e consumo nao avaliados.")
    return resultado
