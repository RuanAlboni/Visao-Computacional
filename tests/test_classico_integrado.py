"""Testes locais sem YOLO e sem checkpoints reais de IA."""
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.alimentos_v2.classico import (
    LIMIARES_PADRAO, classificar_recorte, configurar_limiares, visualizar_recorte
)
from src.alimentos_v2.fluxo import Motor


def imagem_fruit():
    """Fruta sintética isolada em fundo neutro; suficiente para testar o fluxo."""
    a=np.full((144,144,3),235,dtype=np.uint8)
    y,x=np.ogrid[:144,:144]
    m=((x-72)/52)**2+((y-72)/57)**2<1
    a[m]=[210,35,35]
    return a


def caixa(cid, bbox, confianca=.85):
    return SimpleNamespace(cls=np.array([cid]),conf=np.array([confianca]),xyxy=np.array([bbox]))


class FakeDetector:
    def predict(self,**kwargs):
        self.kwargs=kwargs
        return [SimpleNamespace(boxes=[caixa(47,[0,0,144,144]),caixa(51,[144,0,288,144])])]


def motor_fake(modelo=None):
    m=Motor.__new__(Motor)
    m.detector=FakeDetector()
    m.device="cpu"
    m.meta={"classico":"Visao computacional classica", "origem_limiares_classico":"teste"}
    m.nomes={47:"apple",51:"carrot"}
    m.classificador=modelo
    m.usar_classico=True
    m.limiares_classico=LIMIARES_PADRAO.copy()
    return m


def test_regras_classicas_no_recorte():
    a=imagem_fruit()
    info=classificar_recorte(a,"maca",LIMIARES_PADRAO)
    assert info["estado"] in {"fresco","deteriorado"}
    assert 0<=info["pontuacao"]<=1
    overlay,fruta,manchas=visualizar_recorte(a,"maca",LIMIARES_PADRAO["maca"])
    assert overlay.shape==a.shape and fruta.shape==manchas.shape==a.shape[:2]
    assert np.count_nonzero(fruta)>0


def test_classificador_nao_inventa_regras_cenoura():
    r=classificar_recorte(imagem_fruit(),"cenoura",LIMIARES_PADRAO)
    assert r["estado"]=="nao_avaliado"


def test_falha_segmentacao_marcada_inconclusiva():
    r=classificar_recorte(np.zeros((10,10,3),dtype=np.uint8),"maca",LIMIARES_PADRAO)
    assert r["estado"]=="inconclusivo" and "motivo" in r


def test_motor_classe_e_ia_independentes():
    class FalsoProfundo:
        classes=["maca_fresca","maca_deteriorada","banana_fresca"]
        def prever(self,rgb):
            return {"banana_fresca":0.95,"maca_fresca":0.04,"maca_deteriorada":0.01}
    m=motor_fake(FalsoProfundo())
    foto=np.concatenate([imagem_fruit(),imagem_fruit()],axis=1)
    r=m.analisar(foto,agnostic_nms=True)
    fruta,cenoura=r["deteccoes"]
    assert fruta["alimento"]=="maca"
    assert fruta["classico_especie_regras"]=="maca"
    assert fruta["classico"]["estado"] in ("fresco","deteriorado","inconclusivo")
    assert fruta["estado"]=="inconclusivo"  # ResNet discorda da identidade
    assert "classico_aviso" in fruta
    assert cenoura["classico"]["estado"]=="nao_avaliado"
    assert cenoura["estado"]=="nao_avaliado"
    assert m.detector.kwargs["agnostic_nms"] is True
    assert m.detector.kwargs["iou"]==0.5
    json.dumps(r,ensure_ascii=False,allow_nan=False)


def test_limiares_calibrados(tmp_path):
    p=tmp_path/"calibracao.json"
    p.write_text(json.dumps({"limiares":{"maca":0.101,"banana":0.202,"laranja":0.303}}),encoding="utf-8")
    limites,origem=configurar_limiares(p)
    assert limites["banana"]==0.202 and origem==str(p.resolve())
    padrao,origem=configurar_limiares(tmp_path/"ausente.json")
    assert padrao==LIMIARES_PADRAO and "padrao" in origem
