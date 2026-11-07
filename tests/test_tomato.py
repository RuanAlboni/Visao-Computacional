"""Regressoes do bloco 1; nao demonstram acuracia em fotografias reais."""
import numpy as np
import pytest
from PIL import Image, ImageDraw
from src.modules.tomato import TomatoModule
from src.modules.tomato.deteccao import propor_candidatos
from src.classic.tomato_rules import analisar_tomate
from src.schemas.fruit import FruitResult, FruitCondition
from src.utils.anotar import desenhar
from src.pipeline import analisar

def cena(quantidade=2):
    im=Image.new("RGB",(260,160),(225,225,225))
    d=ImageDraw.Draw(im)
    if quantidade>=1: d.ellipse((25,35,100,115),fill=(200,35,25))
    if quantidade>=2: d.ellipse((145,35,220,115),fill=(200,35,25))
    return np.asarray(im)

def test_dois_objetos_separados():
    candidatos,logs=propor_candidatos(cena())
    assert len(candidatos)==2
    assert candidatos[0].bbox[2] < candidatos[1].bbox[0]

def test_sem_objeto_nao_vira_frame():
    candidatos,_=propor_candidatos(cena(0))
    assert candidatos==[]
    assert TomatoModule().detect(cena(0)).total_fruits==0

def test_mascara_obrigatoria():
    with pytest.raises(ValueError,match="Mascara"):
        analisar_tomate(cena())

def test_fundo_nao_altera_descritores():
    a=cena(1).copy(); b=a.copy()
    yy,xx=np.ogrid[:160,:260]
    mask=((xx-62)/35)**2+((yy-75)/35)**2<=1
    b[~mask]=(20,20,20)
    ra=analisar_tomate(a,mascara_objeto=mask)
    rb=analisar_tomate(b,mascara_objeto=mask)
    assert ra["metricas"]==rb["metricas"]

def test_resultado_unico_por_imagem():
    r=TomatoModule().detect(cena())
    assert isinstance(r,FruitResult)
    assert len(r.detections)==2
    assert all(d.condition==FruitCondition.UNKNOWN for d in r.detections)

def test_caixa_xywh():
    m=TomatoModule(); rgb=cena(1)
    detalhes=m.analisar(rgb)
    r=m.para_fruit_result(detalhes,"foto.jpg")
    x1,y1,x2,y2=detalhes[0].bbox; b=r.detections[0].bbox
    assert (b.x,b.y,b.width,b.height)==(x1,y1,x2-x1,y2-y1)
    assert r.image_path=="foto.jpg"

def test_lote_um_resultado_por_entrada():
    r=TomatoModule().detect_batch([cena(0),cena(1),cena(2)])
    assert [x.total_fruits for x in r]==[0,1,2]

def test_caminho_real(tmp_path):
    p=tmp_path/"imagem.png"; Image.fromarray(cena(1)).save(p)
    r=TomatoModule().detect(p)
    assert r.image_path==str(p) and len(r.detections)==1

def test_nada_de_prazo_ou_confianca_inventados():
    r=TomatoModule().analisar(cena(1))[0]
    assert r.dias_restantes is None and r.confianca is None and r.prob_fresco is None
    assert r.apropriado is None

def test_profundo_nao_e_silenciosamente_ignorado():
    with pytest.raises(RuntimeError,match="profundo"):
        TomatoModule(usar_profundo=True)

def test_sem_abordagem_falha():
    with pytest.raises(ValueError): TomatoModule(usar_classico=False)

def test_imagem_invalida():
    with pytest.raises(ValueError): TomatoModule().analisar(np.zeros((30,30),np.uint8))

def test_desenho_nao_altera_original():
    rgb=cena(); copia=rgb.copy()
    out=desenhar(rgb,TomatoModule().analisar(rgb))
    assert np.array_equal(rgb,copia)
    assert out.width==rgb.shape[1] and out.height>rgb.shape[0]

def test_exportacao_json_sem_nan():
    import json
    dados=analisar("tomate",cena(),armazenamento="bancada").para_dict()
    json.dumps(dados,allow_nan=False)
    assert dados["parametros"]["armazenamento"]=="bancada"
    assert dados["resultados"][0]["dias_restantes"] is None

def test_pimentoes_nao_confirmados_como_tomates():
    r=TomatoModule().analisar(cena(1))[0]
    assert not r.detalhes["identidade_confirmada"]
    assert r.veredito=="inconclusivo"


def test_dois_objetos_encostados_sinteticos():
    im=Image.new("RGB",(260,160),(225,225,225))
    d=ImageDraw.Draw(im)
    d.ellipse((35,35,120,120),fill=(200,35,25))
    d.ellipse((110,35,195,120),fill=(200,35,25))
    candidatos,_=propor_candidatos(np.asarray(im))
    assert len(candidatos)==2
