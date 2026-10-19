"""Aplicacao local FruitVision. Execute: streamlit run app.py"""
from __future__ import annotations
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
from src.classico import analisar_classico, sobreposicao, carregar_limiares
from src.constantes import ESPECIES, ROTULOS_ESPECIES, ROTULOS_ESTADOS, rotulo, CLASSES

st.set_page_config(page_title="FruitVision", page_icon="🍎", layout="wide")

BASE = Path(__file__).resolve().parent
MODELOS = BASE / "modelos"
RESULTADOS = BASE / "resultados"

@st.cache_resource(show_spinner="Carregando CNN na memória...")
def _modelo_cache(caminho: str, modificacao: int):
    from src.profundo import carregar_modelo
    return carregar_modelo(caminho)


def _painel_classe(nome: str, resultado: str, detalhe: str, avisos: str = ""):
    cor = "🟢" if resultado == "fresca" else "🟠"
    st.markdown(f"#### {nome}")
    st.markdown(f"### {cor} {ROTULOS_ESTADOS[resultado]}")
    st.caption(detalhe)
    if avisos:
        st.caption(avisos)


def _arquivo_rgb(blob: bytes):
    try:
        with Image.open(io.BytesIO(blob)) as img:
            corrigida = ImageOps.exif_transpose(img).convert("RGB")
            # Protecao contra upload de fotos gigantes (e uso excessivo de memoria).
            corrigida.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            return np.asarray(corrigida)
    except Exception as exc:
        raise ValueError("O arquivo não pôde ser aberto como imagem válida.") from exc


def tela_classificacao():
    st.subheader("Classificar fotografia")
    st.write("Envie uma fotografia de maçã, banana ou laranja. O método clássico precisa saber qual é a espécie; as CNNs preveem as seis classes automaticamente.")
    origem = st.radio("Fonte da imagem", ["Enviar fotografia", "Exemplo artificial"], horizontal=True)
    topo = st.columns([3, 2])
    with topo[0]:
        if origem == "Enviar fotografia":
            imagem = st.file_uploader("Carregar imagem", type=["png", "jpg", "jpeg", "webp", "bmp"])
            blob = imagem.getvalue() if imagem is not None else None
        else:
            arquivos_exemplo = sorted((BASE / "exemplos").glob("*.png"))
            if not arquivos_exemplo:
                st.error("Nenhum exemplo artificial disponível.")
                return
            escolhido = st.selectbox("Imagem demonstrativa (sintética)", arquivos_exemplo,
                                     format_func=lambda p: p.stem.replace("_", " · ").capitalize())
            blob = escolhido.read_bytes()
    with topo[1]:
        especie_inicial = 0
        if origem == "Exemplo artificial":
            especie_inicial = list(ESPECIES).index(escolhido.stem.split("_")[0])
        especie = st.selectbox("Espécie para regras clássicas", list(ESPECIES),
                               index=especie_inicial, format_func=lambda s: ROTULOS_ESPECIES[s])
    if blob is None:
        st.info("Escolha uma imagem para analisar. As regras clássicas funcionam antes do treinamento das CNNs.")
        return
    try:
        rgb = _arquivo_rgb(blob)
    except ValueError as erro:
        st.error(str(erro))
        return
    if origem == "Exemplo artificial":
        st.warning("Esta é uma ilustração simples, usada somente para testar o funcionamento técnico. Não demonstra desempenho em fotografias reais.")
    col_img, col_resp = st.columns([1, 1.5], gap="large")
    with col_img:
        st.image(rgb, caption="Imagem enviada", use_container_width=True)
    with col_resp:
        with st.spinner("Processando abordagem clássica..."):
            try:
                limiares = carregar_limiares(MODELOS / "limiares_classico.json")
                classico = analisar_classico(rgb, especie, limiares[especie])
            except (ValueError, OSError) as erro:
                classico = None
                st.warning(f"Método clássico: classificação indeterminada — {erro}")
        if classico is not None:
            _painel_classe("Visão computacional clássica", classico.estado,
                          f"Índice de deterioração: {classico.pontuacao:.3f} · limiar: {classico.limiar:.3f}",
                          "Índice heurístico, não é probabilidade.")
        checkpoints = [("CNN própria", MODELOS / "cnn.pth"), ("ResNet-18", MODELOS / "resnet18.pth")]
        existentes = [(nome, arquivo) for nome, arquivo in checkpoints if arquivo.exists()]
        if not existentes:
            st.info("Nenhuma CNN treinada encontrada. Após o treinamento, os resultados da IA aparecerão aqui automaticamente.")
        for nome, arquivo in existentes:
            with st.spinner(f"Executando {nome}..."):
                try:
                    from src.profundo import prever_rgb
                    modelo, dispositivo, _ = _modelo_cache(str(arquivo), arquivo.stat().st_mtime_ns)
                    previsao = prever_rgb(rgb, modelo, dispositivo)
                    _painel_classe(nome, previsao["estado"],
                                  f"Espécie prevista: {ROTULOS_ESPECIES[previsao['especie']]} · "
                                  f"Probabilidade estimada de deterioração: {previsao['probabilidade_deteriorada']:.1%}",
                                  "Probabilidades não necessariamente calibradas.")
                    with st.expander(f"Probabilidades por classe — {nome}"):
                        df = pd.DataFrame({"classe": [rotulo(k) for k in CLASSES],
                                           "probabilidade": [previsao["probabilidades"][k] for k in CLASSES]})
                        st.bar_chart(df.set_index("classe"))
                    if previsao["especie"] != especie:
                        st.caption(f"Atenção: a espécie escolhida para o clássico é {ROTULOS_ESPECIES[especie]}, "
                                   f"mas {nome} previu {ROTULOS_ESPECIES[previsao['especie']]}. Confira a entrada.")
                except Exception as erro:
                    st.error(f"Não foi possível executar {nome}: {erro}")
    if classico is not None:
        st.divider()
        st.subheader("Por que o método clássico tomou essa decisão?")
        imgs = st.columns(3)
        with imgs[0]:
            st.image(classico.imagem, caption="Imagem analisada", use_container_width=True)
        with imgs[1]:
            st.image((classico.mascara_fruta.astype(np.uint8) * 255), caption="Máscara da fruta", use_container_width=True, clamp=True)
        with imgs[2]:
            st.image(sobreposicao(classico), caption="Vermelho: manchas · Amarelo: textura suspeita", use_container_width=True)
        a, b, c, d = st.columns(4)
        a.metric("Área com manchas", f"{classico.fracao_manchas:.1%}")
        b.metric("Textura irregular", f"{classico.fracao_textura:.1%}")
        c.metric("Entropia LBP", f"{classico.lbp_entropia:.2f}")
        d.metric("Contraste GLCM", f"{classico.glcm_contraste:.2f}")
        st.caption(classico.aviso + " LBP e GLCM são medidas descritivas; o índice de decisão utiliza as proporções de manchas e textura.")


def tela_resultados():
    st.subheader("Avaliação experimental")
    arquivo = RESULTADOS / "resumo.json"
    if not arquivo.exists():
        st.info("Ainda não há resultados. Execute 'python main.py avaliar ...' para gerar as métricas no conjunto de teste.")
        return
    try:
        resumo = json.loads(arquivo.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        st.error("Arquivo de métricas inválido.")
        return
    st.caption(f"Imagens totais de teste: {resumo.get('amostras_teste', 'não informado')}. Todas as métricas foram calculadas offline.")
    linhas = []
    for modelo, dados in resumo.items():
        if modelo == "amostras_teste" or not isinstance(dados, dict):
            continue
        metrica = dados.get("geral", dados.get("condicao_geral", {}))
        if not metrica.get("n"):
            continue
        linhas.append({"Método": modelo, "Imagens avaliadas": metrica["n"],
                       "Acurácia": metrica["acuracia"], "Precisão (deteriorada)": metrica["precisao_deteriorada"],
                       "Revocação (deteriorada)": metrica["revocacao_deteriorada"], "F1 macro": metrica["f1_macro"]})
    if linhas:
        dataframe = pd.DataFrame(linhas).set_index("Método")
        st.dataframe(dataframe.style.format({x: "{:.1%}" for x in dataframe.columns if x not in ["Imagens avaliadas"]}),
                     use_container_width=True)
    if "classico" in resumo:
        cobertura = resumo["classico"].get("cobertura", 0)
        st.caption(f"Cobertura do classificador clássico (sem falha de segmentação): {cobertura:.1%}. "
                   "Suas métricas consideram apenas imagens classificáveis; consulte o CSV para as falhas.")
    matrizes = sorted(RESULTADOS.glob("confusao_*.png"))
    if matrizes:
        st.markdown("#### Matrizes de confusão")
        for inicio in range(0, len(matrizes), 2):
            cols = st.columns(2)
            for col, path in zip(cols, matrizes[inicio:inicio+2]):
                with col:
                    st.image(str(path), caption=path.stem.replace("confusao_", ""), use_container_width=True)
    st.markdown("#### Evolução do treinamento")
    for nome in ["resnet18", "cnn"]:
        arquivo_hist = MODELOS / f"{nome}.json"
        if arquivo_hist.exists():
            try:
                hist = json.loads(arquivo_hist.read_text(encoding="utf-8"))
                df = pd.DataFrame({"Época": [x["epoca"] for x in hist],
                                   "Treino": [x["treino"]["f1_macro"] for x in hist],
                                   "Validação": [x["validacao"]["f1_macro"] for x in hist]}).set_index("Época")
                st.write(f"**{nome} — F1 macro (seis classes)**")
                st.line_chart(df)
            except (ValueError, KeyError):
                st.warning(f"Histórico de {nome} não pôde ser lido.")


def tela_sobre():
    st.subheader("Como funciona")
    st.write("**Clássico:** segmentação da fruta, análise de cor HSV/CIELAB, manchas, resíduo de textura local, métricas LBP/GLCM e limiar por espécie calibrado na validação.")
    st.write("**IA:** duas arquiteturas alternativas de classificação em seis classes: CNN própria e ResNet-18. Treinamento supervisionado em PyTorch, salvando o melhor checkpoint na validação.")
    st.write("**Comparação:** as três abordagens predizem o estado visual 'fresca' ou 'deteriorada'. A IA também prevê a espécie. O clássico recebe a espécie correta como informação adicional, portanto a comparação não é perfeitamente equivalente no reconhecimento da fruta.")
    st.info("A ferramenta detecta sinais visuais; não determina segurança microbiológica nem certifica se um alimento é seguro para consumo.")
    st.markdown("Veja o `README.md` para instruções de obtenção dos dados, treinamento na RTX 5070 e avaliação.")


st.title("🍎 FruitVision")
st.caption("Classificação visual de frutas · OpenCV + Deep Learning · execução local")
aba_entrada, aba_metricas, aba_sobre = st.tabs(["Classificação", "Experimentos", "Sobre o projeto"])
with aba_entrada:
    tela_classificacao()
with aba_metricas:
    tela_resultados()
with aba_sobre:
    tela_sobre()
