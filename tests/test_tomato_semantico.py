"""Testes do adaptador com saidas controladas. Nao medem acuracia de pesos reais."""
from types import SimpleNamespace
import numpy as np
import pytest
from src.modules.tomato.semantico import DetectorTomate,DetectorIncompativel
from src.modules.tomato.localizacao import executar
from src.modules.tomato.vis_localizacao import desenhar

def caixa(cid=0,conf=0.9,xy=(1,2,20,25)):
    return SimpleNamespace(cls=np.array([cid]),conf=np.array([conf]),xyxy=np.array([xy],dtype=float))

class Fake:
    task="detect"
    names={0:"Tomato",1:"Bell pepper"}
    def __init__(self,boxes=None): self.boxes=[caixa()] if boxes is None else boxes
    def predict(self,**kwargs):
        self.kwargs=kwargs
        return [SimpleNamespace(boxes=self.boxes)]

def rgb(): return np.full((40,50,3),(200,30,10),np.uint8)

def test_filtra_pimentao_mesmo_se_backend_devolver():
    d=DetectorTomate(Fake([caixa(0),caixa(1)]))
    assert len(d.prever(rgb()))==1

def test_rgb_convertido_bgr():
    f=Fake();DetectorTomate(f).prever(rgb())
    assert f.kwargs["source"][0,0].tolist()==[10,30,200]
    assert f.kwargs["classes"]==[0]

def test_rejeita_modelo_sem_tomate():
    f=Fake();f.names={0:"apple",1:"pepper"}
    with pytest.raises(DetectorIncompativel): DetectorTomate(f)

def test_rejeita_classificador():
    f=Fake();f.task="classify"
    with pytest.raises(DetectorIncompativel): DetectorTomate(f)

def test_classe_nao_aceita_por_substring():
    f=Fake();f.names={0:"tomato_leaf"}
    with pytest.raises(DetectorIncompativel): DetectorTomate(f)

def test_clip_e_descarte_caixa_invalida():
    f=Fake([caixa(xy=(-4,-8,200,100)),caixa(xy=(20,20,10,10))])
    saida=DetectorTomate(f).prever(rgb())
    assert len(saida)==1 and saida[0].bbox==(0,0,50,40)

def test_limiar_confianca():
    assert DetectorTomate(Fake([caixa(conf=0.1)])).prever(rgb(),conf=0.35)==[]

def test_modelo_indisponivel_nao_e_zero_medido():
    d=executar(rgb(),executar_classico=False)
    assert d["status_modelo"]=="indisponivel" and d["deteccoes_do_modelo"]==[]
    assert d["estado"]=="nao_avaliado" and d["dias_restantes"] is None

def test_zero_predicoes_de_modelo_executado():
    d=executar(rgb(),DetectorTomate(Fake([])),False)
    assert d["status_modelo"]=="executado" and len(d["deteccoes_do_modelo"])==0

def test_erro_inferencia_nao_cai_no_classico():
    class Erro:
        def prever(self,*args,**kwargs): raise RuntimeError("erro controlado")
    d=executar(rgb(),Erro(),False)
    assert d["status_modelo"]=="erro" and "modelo" in d["erros"]

def test_id_novo_mesma_imagem_hash_igual():
    a=executar(rgb(),executar_classico=False);b=executar(rgb(),executar_classico=False)
    assert a["id"]!=b["id"] and a["imagem_sha256"]==b["imagem_sha256"]
    assert a["id"] in a["logs"][0]

def test_renderiza_apenas_fonte_selecionada():
    d=executar(rgb(),DetectorTomate(Fake()),False)
    d["candidatos_classicos"]=[{"bbox":[30,10,45,30]}]
    a=np.asarray(desenhar(rgb(),d,"deteccoes_do_modelo"))
    b=np.asarray(desenhar(rgb(),d,"candidatos_classicos"))
    assert a.shape==b.shape and not np.array_equal(a,b)

def test_json_valido():
    import json
    d=executar(rgb(),DetectorTomate(Fake()),False)
    json.dumps(d,allow_nan=False)
    assert d["deteccoes_do_modelo"][0]["classe_nome"]=="Tomato"
