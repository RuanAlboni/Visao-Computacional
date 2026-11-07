"""Apresentacao de previsoes; nao altera deteccoes, scores ou IDs."""
from collections import Counter
import numpy as np
from PIL import Image,ImageDraw,ImageFont

CORES={"banana":(250,204,21),"maca":(232,90,170),"laranja":(255,145,35),
       "cenoura":(50,200,240),"brocolis":(85,210,145)}
NOMES={"banana":"Banana","maca":"Maçã","laranja":"Laranja","cenoura":"Cenoura","brocolis":"Brócolis"}

def resumo(dados):
    contagens=Counter(o["alimento"] for o in dados["deteccoes"])
    selecionados=dados.get("parametros",{}).get("alimentos",list(CORES))
    produtos=list(dict.fromkeys(list(selecionados)+list(contagens)))
    return [{"produto":p,"nome":NOMES.get(p,p),"deteccoes":contagens[p],
             "ids":[o["id_objeto"] for o in dados["deteccoes"] if o["alimento"]==p]} for p in produtos]

def fonte(tamanho):
    try:return ImageFont.truetype("DejaVuSans.ttf",tamanho)
    except OSError:return ImageFont.load_default()

def anotar(rgb,dados):
    base=Image.fromarray(np.asarray(rgb)).convert("RGB")
    largura=max(560,base.width)
    linhas=resumo(dados);colunas=2;passo=29
    topo=78+((len(linhas)+colunas-1)//colunas)*passo
    im=Image.new("RGB",(largura,base.height+topo),(18,22,30))
    im.paste(base,(0,topo));d=ImageDraw.Draw(im)
    f=fonte(16);pequena=fonte(14)
    d.text((10,8),"ID "+dados["id"],font=pequena,fill="white")
    d.text((10,31),f"{len(dados['deteccoes'])} detecções — não equivale a unidades confirmadas",font=pequena,fill="white")
    for indice,linha in enumerate(linhas):
        x=10+(indice%colunas)*(largura//colunas);y=65+(indice//colunas)*passo
        cor=CORES.get(linha["produto"],(190,190,190))
        d.rectangle((x,y,x+14,y+14),fill=cor)
        d.text((x+22,y-3),f"{linha['nome']}: {linha['deteccoes']}",font=f,fill="white")
    for obj in dados["deteccoes"]:
        x1,y1,x2,y2=map(int,obj["bbox"]);y1+=topo;y2+=topo
        cor=CORES.get(obj["alimento"],(190,190,190))
        if x2<=x1 or y2<=y1:continue
        d.rectangle((x1,y1,x2-1,y2-1),outline=cor,width=4)
    ocupados=[]
    for obj in dados["deteccoes"]:
        x1,y1,x2,y2=map(int,obj["bbox"]);y1+=topo;y2+=topo
        cor=CORES.get(obj["alimento"],(190,190,190));texto=f"#{obj['id_objeto']}"
        b=d.textbbox((0,0),texto,font=f);ww=b[2]-b[0]+12;hh=b[3]-b[1]+10
        candidatos=[(x1,y1),(x2-ww,y1),(x1,y2-hh),(x2-ww,y2-hh)]
        candidatos += [(x1,y1+i*(hh+3)) for i in range(1,5)]
        def limitar(p):return (max(0,min(p[0],im.width-ww)),max(topo,min(p[1],im.height-hh)))
        candidatos=[limitar(p) for p in candidatos]
        def area_sobreposta(p):
            x,y=p
            return sum(max(0,min(x+ww,c)-max(x,a))*max(0,min(y+hh,e)-max(y,b)) for a,b,c,e in ocupados)
        x,y=min(candidatos,key=area_sobreposta)
        ocupados.append((x,y,x+ww,y+hh))
        if x!=x1 or y!=y1:d.line((x1,y1,x+ww//2,y+hh//2),fill=cor,width=2)
        d.rectangle((x,y,x+ww,y+hh),fill=cor,outline="black",width=1)
        d.text((x+6,y+5-b[1]),texto,font=f,fill="black")
    return im

def resumo_painel(dados):
    import streamlit as st
    st.subheader("Detecções por alimento")
    linhas=resumo(dados)
    for inicio in range(0,len(linhas),5):
        grupo=linhas[inicio:inicio+5];cols=st.columns(len(grupo))
        for col,linha in zip(cols,grupo):
            r,g,b=CORES.get(linha["produto"],(190,190,190))
            with col:
                st.markdown(f'<span style="display:inline-block;width:14px;height:14px;background:rgb({r},{g},{b});border-radius:3px"></span>',unsafe_allow_html=True)
                st.metric(linha["nome"],linha["deteccoes"])
                st.caption("IDs: "+(", ".join(f"#{i}" for i in linha["ids"]) or "nenhum"))
    st.caption(f"Total: {len(dados['deteccoes'])} caixas previstas. Caixas sobrepostas podem corresponder ao mesmo objeto ou a um grupo. Zero detecções não comprova ausência.")
    modelos=dados.get("modelos",{})
    if not modelos.get("classificador"):
        st.info("Classificador de estado não carregado nesta análise. As cores indicam alimento, não fresco/deteriorado.")
    else:
        st.caption("Classificador carregado: "+str(modelos["classificador"]))

def galeria(rgb,dados):
    import streamlit as st
    a=np.asarray(rgb)
    st.caption("Confira a identidade de cada previsão. Estes recortes usam as caixas originais, sem alterar a contagem.")
    for inicio in range(0,len(dados["deteccoes"]),3):
        grupo=dados["deteccoes"][inicio:inicio+3];cols=st.columns(len(grupo))
        for col,obj in zip(cols,grupo):
            x1,y1,x2,y2=map(int,obj["bbox"])
            x1=max(0,x1);y1=max(0,y1);x2=min(a.shape[1],x2);y2=min(a.shape[0],y2)
            with col:
                st.markdown(f"#{obj['id_objeto']} — {NOMES.get(obj['alimento'],obj['alimento'])}")
                if x2>x1 and y2>y1:st.image(a[y1:y2,x1:x2],use_container_width=True)
                st.caption(f"Score do detector: {obj['detection_score']:.3f}")
                st.caption("Estado: "+str(obj.get("estado","nao_avaliado")))
                if "classe_prevista" in obj:st.caption("Classe: "+str(obj["classe_prevista"]))
                if "score" in obj:st.caption(f"Score do classificador: {obj['score']:.3f}")
