
"""Interface da revisao de localizacao; sem classificacao de consumo."""
import io
import json
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import streamlit as st
from src.modules.tomato.localizacao import executar
from src.modules.tomato.semantico import DetectorTomate
from src.modules.tomato.vis_localizacao import desenhar

@st.cache_resource
def carregar_detector(caminho,tamanho,mtime,device):
    return DetectorTomate.carregar(caminho,device=device)

@st.cache_resource
def logger():
    Path("logs").mkdir(exist_ok=True)
    obj=logging.getLogger("tomato_localizacao");obj.setLevel(logging.INFO);obj.propagate=False
    if not obj.handlers:
        h=TimedRotatingFileHandler("logs/tomato_localizacao.log",when="midnight",backupCount=7,encoding="utf-8")
        h.setFormatter(logging.Formatter("%(asctime)s %(message)s"));obj.addHandler(h)
    return obj

def main():
    st.set_page_config(page_title="Smart Fridge Vision — localizacao",layout="wide")
    st.title("Smart Fridge Vision — validar localizacao de tomates")
    st.info("Sem classificacao de estado ou consumo nesta etapa. Contagens do modelo sao predicoes, nao confirmacoes.")
    with st.sidebar:
        with st.form("localizacao"):
            imagem=st.file_uploader("Imagem",type=["jpg","jpeg","png","webp","bmp"])
            pesos=st.text_input("Checkpoint local confiavel (.pt)",value="models/tomato_detector.pt")
            usar_modelo=st.checkbox("Executar detector semantico",value=True)
            usar_classico=st.checkbox("Executar candidatos classicos para comparar",value=True)
            device=st.selectbox("Dispositivo",["cpu","cuda"])
            conf=st.slider("Limiar do detector",0.05,0.95,0.35,0.05)
            iou=st.slider("IoU de supressao do detector (nao e metrica de avaliacao)",0.10,0.90,0.50,0.05)
            enviado=st.form_submit_button("Analisar")
    if enviado:
        st.session_state.pop("tomato_localizacao",None)
        if imagem is None: st.error("Envie uma imagem.")
        else:
            try:
                with Image.open(imagem) as origem:
                    if origem.width*origem.height>20_000_000: raise ValueError("Limite de 20 milhoes de pixels.")
                    rgb=np.asarray(ImageOps.exif_transpose(origem).convert("RGB"))
                detector=None;erro_detector=None;meta={}
                if usar_modelo:
                    try:
                        p=Path(pesos).expanduser().resolve()
                        if not p.is_file(): raise FileNotFoundError("Pesos locais nao encontrados; nenhum modelo foi baixado.")
                        stat=p.stat();detector=carregar_detector(str(p),stat.st_size,stat.st_mtime_ns,device)
                        meta={"caminho":detector.caminho,"sha256":detector.sha256,
                              "classes":detector.nomes,"classes_filtradas":detector.ids,"device":device}
                    except Exception as e: erro_detector=f"{type(e).__name__}: {e}"
                with st.spinner("Processando..."):
                    dados=executar(rgb,detector,usar_classico,conf,iou,meta)
                dados["arquivo"]=imagem.name
                if not usar_modelo: dados["status_modelo"]="nao_solicitado"
                if erro_detector:
                    dados["erros"]["carregamento_modelo"]=erro_detector
                    dados["logs"].append("Modelo indisponivel: "+erro_detector)
                dados["logs"].append("status_modelo_final="+dados["status_modelo"])
                logger().info(json.dumps(dados,ensure_ascii=False,allow_nan=False))
                st.session_state["tomato_localizacao"]={"rgb":rgb,"dados":dados}
            except Exception as e:
                st.error(f"Analise nao concluida: {e}");logger().exception("Falha de analise")
    salvo=st.session_state.get("tomato_localizacao")
    if not salvo: st.caption("Selecione uma imagem e clique em Analisar.");return
    dados=salvo["dados"];rgb=salvo["rgb"]
    st.caption(f"ID {dados['id']} | {dados['arquivo']} | {dados['duracao_s']:.3f}s de processamento (sem carregamento)")
    a,b=st.columns(2)
    a.metric("Candidatos classicos",len(dados["candidatos_classicos"]) if dados["status_classico"]=="executado" else "Nao executado")
    b.metric("Tomates previstos pelo modelo",len(dados["deteccoes_do_modelo"]) if dados["status_modelo"]=="executado" else "Indisponivel")
    for chave,valor in dados["erros"].items(): st.warning(f"{chave}: {valor}")
    fonte=st.radio("Caixas a exibir",["deteccoes_do_modelo","candidatos_classicos"],horizontal=True)
    anotada=desenhar(rgb,dados,fonte)
    a,b=st.columns(2);a.image(rgb,caption="Original",use_container_width=True)
    b.image(anotada,caption=fonte+" — ID "+dados["id"],use_container_width=True)
    if fonte=="deteccoes_do_modelo" and dados["status_modelo"]!="executado":
        st.warning("Sem predicao semantica nesta analise. Nao interprete isso como zero tomates.")
    st.dataframe([{"numero":i,**obj} for i,obj in enumerate(dados[fonte],1)],use_container_width=True)
    st.subheader("Logs");st.code("\n".join(dados["logs"]),language="text")
    buff=io.BytesIO();anotada.save(buff,format="PNG")
    st.download_button("Baixar imagem selecionada",buff.getvalue(),f"{dados['id']}_{fonte}.png","image/png")
    st.download_button("Baixar JSON completo",json.dumps(dados,ensure_ascii=False,indent=2,allow_nan=False),f"{dados['id']}.json","application/json")
