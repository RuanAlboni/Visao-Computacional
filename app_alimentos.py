"""Painel unificado. Executar: python -m streamlit run app_alimentos.py."""
from src.alimentos_v2.visual import anotar, resumo_painel, galeria
import io,json,logging,shlex,os,subprocess
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler
import numpy as np
from PIL import Image,ImageOps,ImageDraw,ImageFont
import streamlit as st
from src.alimentos_v2.catalogo import ALIMENTOS
from src.alimentos_v2.fluxo import Motor

@st.cache_resource
def carregar(det,det_stat,estado,estado_stat,device,usar_classico,limiares_path,limiares_stat):
    return Motor(det,estado,device,usar_classico=usar_classico,limiares_classico=limiares_path)

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


def escapar_parametro(valor):
    # PowerShell/cmd e shells POSIX usam regras diferentes de aspas.
    return subprocess.list2cmdline([str(valor)]) if os.name == "nt" else shlex.quote(str(valor))


st.set_page_config(page_title="Smart Fridge Vision — alimentos",layout="wide")
st.title("Smart Fridge Vision — painel multialimento")
st.caption("YOLO localiza os alimentos; visão computacional clássica e aprendizado profundo avaliam independentemente maçãs, bananas e laranjas.")
st.warning("Estado previsto e aparencia nao confirmam seguranca alimentar. Prazo restante nao estimado.")
aba,treino=st.tabs(["Analise","Treinamento multiclasse"])
with treino:
    st.write("Treinamento real pelo terminal; esta aba gera o comando, nao inicia treinamento em segundo plano.")
    dados_t=st.text_input("Dataset com train/val/test ou manifesto.csv",value="data/alimentos")
    saida_t=st.text_input("Checkpoint de saida",value="models/novo_multiclasse.pth")
    ep=st.number_input("Épocas máximas",min_value=1,max_value=500,value=25)
    paciencia=st.number_input("Paciência (early stopping)",min_value=1,max_value=100,value=6)
    arquitetura=st.selectbox("Arquitetura para treino",["resnet18","cnn"])
    dispositivo_treino=st.selectbox("Dispositivo de treino",["auto","cuda","cpu"])
    comando=(f"python scripts/treinar_multiclasse.py --dados {escapar_parametro(dados_t)} "
             f"--saida {escapar_parametro(saida_t)} --arquitetura {arquitetura} --epocas {ep} "
             f"--paciencia {paciencia} --device {dispositivo_treino}")
    st.code(comando,language="bash")
    st.caption("Calibração clássica (somente val/):")
    st.code(f"python scripts/calibrar_classico.py --dados {escapar_parametro(dados_t)} "
            f"--saida models/limiares_classico_novos.json",language="bash")
    st.caption("Aceita datasets organizados em pastas train/val ou por manifesto CSV. Mantenha treino, validação e teste independentes.")
with aba:
    with st.form("analise_alimentos"):
        arquivo=st.file_uploader("Imagem",type=["jpg","jpeg","png","webp"])
        modo=st.radio("Escopo",["Todos os alimentos disponiveis","Selecionar alimentos"],horizontal=True)
        selecionados=st.multiselect("Filtro (nao muda o modelo)",list(ALIMENTOS),default=["cenoura"])
        pesos=st.text_input("Detector local",value="models/yolo11n.pt")
        usar_estado=st.checkbox("Usar classificador IA (ResNet/CNN)",value=False)
        pesos_estado=st.text_input("Checkpoint do Classificador",value="models/alimentos_multiclasse.pth")
        usar_classico=st.checkbox("Usar classificador clássico (maçã, banana e laranja)",value=True)
        limiares_classico=st.text_input("Arquivo de calibração clássico (opcional)",value="models/limiares_classico.json",disabled=not usar_classico,help="Selecione o arquivo JSON contendo os limiares calibrados. Se não estiver disponível, serão utilizados os limiares heurísticos iniciais.")
        agnostic_nms=st.checkbox("NMS independente da classe (remover duplicatas)",value=True)
        device=st.selectbox("Dispositivo",["cpu","cuda"])
        conf=st.slider("Score minimo do detector",0.05,0.95,0.35,0.05)
        limiar=st.slider("Score minimo para triagem visual de estado",0.10,0.95,0.60,0.05)
        enviar=st.form_submit_button("Analisar")
    if enviar:
        st.session_state.pop("resultado_alimentos",None)
        try:
            if arquivo is None:raise ValueError("Escolha uma imagem")
            det,ds=assinatura(pesos);est,es=assinatura(pesos_estado) if usar_estado else (None,None)
            caminho_lim=Path(limiares_classico).expanduser().resolve() if usar_classico and limiares_classico.strip() else None
            ls=(caminho_lim.stat().st_size,caminho_lim.stat().st_mtime_ns) if caminho_lim and caminho_lim.is_file() else None
            with Image.open(arquivo) as im:
                if im.width*im.height>20_000_000:raise ValueError("Limite: 20 milhoes de pixels")
                rgb=np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
            with st.spinner("Executando detector e classificadores..."):
                motor=carregar(det,ds,est,es,device,usar_classico,str(caminho_lim) if caminho_lim else None,ls)
                resultado=motor.analisar(rgb,None if modo.startswith("Todos") else selecionados,conf,limiar,agnostic_nms=agnostic_nms)
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
        st.subheader("Comparação por detecção")
        linhas=[]
        for obj in resultado["deteccoes"]:
            cl=obj.get("classico",{})
            estado_ia=obj.get("estado","nao_avaliado")
            estado_cl=cl.get("estado","nao_avaliado")
            linhas.append({
                "ID":obj["id_objeto"], "YOLO":obj["alimento"],
                "Detecção YOLO":round(obj["detection_score"],3),
                "IA — estado":estado_ia,
                "IA — classe":obj.get("classe_prevista","—"),
                "IA — prob. classe":round(obj["score"],3) if "score" in obj else None,
                "Clássico — estado":estado_cl,
                "Clássico — pontuação":cl.get("pontuacao"),
                "Clássico — limiar":cl.get("limiar"),
                "Discordância no estado":("SIM" if estado_ia!=estado_cl else "não") if estado_ia in {"fresco","deteriorado"} and estado_cl in {"fresco","deteriorado"} else "—",
                "Avisos":obj.get("classico_aviso",cl.get("motivo", "")),
            })
        st.dataframe(linhas,use_container_width=True,hide_index=True)
        if resultado["modelos"].get("classico") and "padrao" in resultado["modelos"].get("origem_limiares_classico",""):
            st.warning("O classificador clássico está utilizando limiares iniciais, não calibrados. Execute o script de calibração e selecione o JSON gerado para utilizar os limiares otimizados.")
        st.subheader("Log / relatorio da mesma analise")
        txt=json.dumps(resultado,ensure_ascii=False,indent=2,allow_nan=False);st.code(txt,language="json")
        buff=io.BytesIO();anotada.save(buff,format="PNG")
        st.download_button("Baixar imagem",buff.getvalue(),resultado["id"]+".png","image/png")
        st.download_button("Baixar JSON",txt,resultado["id"]+".json","application/json")
