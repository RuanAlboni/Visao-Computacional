from types import SimpleNamespace
import numpy as np
import pytest
from src.alimentos_v2.catalogo import decompor,validar_classes,escolher_estado
from src.alimentos_v2.fluxo import Motor

def test_classes_genero():
    assert decompor("maca_deteriorada")==("maca","deteriorado")
    assert decompor("cenoura_fresca")==("cenoura","fresco")

def test_classe_desconhecida_rejeitada():
    with pytest.raises(ValueError):decompor("tomate_fresco")

def test_duplicadas_rejeitadas():
    with pytest.raises(ValueError):validar_classes(["maca_fresca","maca_fresca"])

def test_discordancia_nao_aprova():
    r=escolher_estado({"banana_fresca":0.9,"maca_fresca":0.1},"maca")
    assert r["estado"]=="inconclusivo"

def test_score_baixo_inconclusivo():
    assert escolher_estado({"maca_fresca":0.51,"maca_deteriorada":0.49},"maca")["estado"]=="inconclusivo"

def test_estado_multiclasse():
    assert escolher_estado({"cenoura_fresca":0.1,"cenoura_deteriorada":0.9},"cenoura")["estado"]=="deteriorado"

class DetectorFake:
    def predict(self,**kwargs):
        self.kwargs=kwargs
        caixa=SimpleNamespace(cls=np.array([51]),conf=np.array([0.8]),xyxy=np.array([[4,5,25,30]]))
        return [SimpleNamespace(boxes=[caixa])]

def motor():
    m=Motor.__new__(Motor);m.detector=DetectorFake();m.device="cpu";m.meta={}
    m.nomes={51:"carrot",47:"apple"};m.classificador=None;return m

def test_cenoura_detectada_sem_estado_inventado():
    m=motor();rgb=np.full((40,50,3),(200,80,10),np.uint8)
    d=m.analisar(rgb,["cenoura"])
    assert d["total_previsto"]==1
    assert d["deteccoes"][0]["estado"]=="nao_avaliado"
    assert d["deteccoes"][0]["dias_restantes"] is None
    assert m.detector.kwargs["source"][0,0].tolist()==[10,80,200]
    assert m.detector.kwargs["classes"]==[51]

def test_checkpoint_frutas_nao_classifica_cenoura():
    m=motor()
    class Frutas:
        classes=["maca_fresca","maca_deteriorada"]
        def prever(self,rgb):raise AssertionError("Nao deveria classificar cenoura")
    m.classificador=Frutas()
    d=m.analisar(np.zeros((40,50,3),np.uint8),["cenoura"])
    assert d["deteccoes"][0]["estado"]=="nao_avaliado"

def test_ids_e_json():
    import json
    m=motor();rgb=np.zeros((40,50,3),np.uint8)
    a=m.analisar(rgb);b=m.analisar(rgb)
    assert a["id"]!=b["id"] and a["imagem_sha256"]==b["imagem_sha256"]
    json.dumps(a,allow_nan=False)

def test_selecao_vazia_falha():
    with pytest.raises(ValueError):motor().analisar(np.zeros((40,50,3),np.uint8),[])
