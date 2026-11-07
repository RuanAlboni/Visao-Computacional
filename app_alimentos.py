from src.alimentos_v2.visual import anotar, resumo_painel, galeria
"""Painel unificado. Executar: python -m streamlit run app_alimentos.py."""
import io,json,logging,shlex
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
import streamlit as st
from src.alimentos_v2.catalogo import ALIMENTOS
from src.alimentos_v2.fluxo import Motor

@st.cache_resource
def carregar(det,det_stat,estado,estado_stat,device):
    return Motor(det,estado,device)

@st.cache_resource
def logger():
    Path("logs").mkdir(exist_ok=True)
    l=logging.getLogger("alimentos_v2");l.setLevel(logging.INFO);l.propagate=False
    if not l.handlers:
        h=TimedRotatingFileHandler("logs/alimentos_v2.log",when="midnight",backupCount=7,encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"));l.addHandler(h)
    return l

def assinatura(caminho):
    p=Path(caminho).expanduser().resolve()
    if not p.is_file():raise FileNotFoundError(str(p))
    s=p.stat();return str(p),(s.st_size,s.st_mtime_ns)


st.set_page_config(page_title="Smart Fridge Vision — alimentos",layout="wide")
st.title("Smart Fridge Vision — painel multialimento")
st.caption("Um detector compartilhado e um classificador multiclasse opcional; sem selecionar uma rede por fruta.")
st.warning("Estado previsto e aparencia nao confirmam seguranca alimentar. Prazo restante nao estimado.")
aba,treino=st.tabs(["Analise","Treinamento multiclasse"])
with treino:
    st.write("Treinamento real pelo terminal; esta aba gera o comando, nao inicia treinamento em segundo plano.")
    dados_t=st.text_input("Dataset com train/val/test",value="data/alimentos")
    saida_t=st.text_input("Checkpoint de saida",value="models/alimentos_multiclasse.pth")
    ep=st.number_input("Epocas",min_value=1,max_value=500,value=15)
    st.code(f"python scripts/treinar_multiclasse.py --dados {shlex.quote(dados_t)} --saida {shlex.quote(saida_t)} --epocas {ep}",language="bash")
    st.caption("Cada classe e uma pasta: maca_fresca, maca_deteriorada, banana_fresca etc. Cenoura exige cenoura_fresca e cenoura_deteriorada no treinamento. Nao separe fotos do mesmo fruto entre treino e validacao.")
with aba:
    with st.form("analise_alimentos"):
        arquivo=st.file_uploader("Imagem",type=["jpg","jpeg","png","webp"])
        modo=st.radio("Escopo",["Todos os alimentos disponiveis","Selecionar alimentos"],horizontal=True)
        selecionados=st.multiselect("Filtro (nao muda o modelo)",list(ALIMENTOS),default=["cenoura"])
        pesos=st.text_input("Detector local",value="models/yolo11n.pt")
        usar_estado=st.checkbox("Usar classificador multiclasse de estado",value=False)
        pesos_estado=st.text_input("Checkpoint de Ruan ou treinado aqui",value="models/alimentos_multiclasse.pth")
        device=st.selectbox("Dispositivo",["cpu","cuda"])
        conf=st.slider("Score minimo do detector",0.05,0.95,0.35,0.05)
        limiar=st.slider("Score minimo para triagem visual de estado",0.10,0.95,0.60,0.05)
        enviar=st.form_submit_button("Analisar")
    if enviar:
        st.session_state.pop("resultado_alimentos",None)
        try:
            if arquivo is None:raise ValueError("Escolha uma imagem")
            det,ds=assinatura(pesos);est,es=assinatura(pesos_estado) if usar_estado else (None,None)
            with Image.open(arquivo) as im:
                if im.width*im.height>20_000_000:raise ValueError("Limite: 20 milhoes de pixels")
                rgb=np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
            with st.spinner("Executando redes..."):
                motor=carregar(det,ds,est,es,device)
                resultado=motor.analisar(rgb,None if modo.startswith("Todos") else selecionados,conf,limiar)
            resultado["arquivo"]=arquivo.name
            logger().info(json.dumps(resultado,ensure_ascii=False,allow_nan=False))
            st.session_state["resultado_alimentos"]={"rgb":rgb,"dados":resultado}
        except Exception as e:
            st.error(f"Analise nao executada: {type(e).__name__}: {e}");logger().exception("Falha de analise")
    salvo=st.session_state.get("resultado_alimentos")
    if salvo:
        resultado=salvo["dados"];rgb=salvo["rgb"]
        st.caption(f"ID {resultado['id']} | {resultado['arquivo']} | {resultado['duracao_s']}s (sem carregar pesos)")
        resumo_painel(resultado)
        anotada=anotar(rgb,resultado);a,b=st.columns(2)
        a.image(rgb,caption="Original",use_container_width=True);b.image(anotada,caption="Previsoes",use_container_width=True)
        with st.expander("Conferir cada detecção — recortes e scores", expanded=True):
            galeria(rgb,resultado)
        st.dataframe([{k:v for k,v in obj.items() if k!="probabilidades"} for obj in resultado["deteccoes"]],use_container_width=True)
        st.subheader("Log / relatorio da mesma analise")
        txt=json.dumps(resultado,ensure_ascii=False,indent=2,allow_nan=False);st.code(txt,language="json")
        buff=io.BytesIO();anotada.save(buff,format="PNG")
        st.download_button("Baixar imagem",buff.getvalue(),resultado["id"]+".png","image/png")
        st.download_button("Baixar JSON",txt,resultado["id"]+".json","application/json")
