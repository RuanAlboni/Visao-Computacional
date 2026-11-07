"""Calibracao por validacao; comparacao no teste reservado."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, f1_score, classification_report)
from src.modules.apple.config import ESPECIES, CLASSES, decompor_classe
from src.data.kaggle_loader import carregar_manifesto
from src.classic._classico_original import analisar_classico, abrir_rgb, carregar_limiares, LIMIARES_PADRAO


def _metricas(verdade, predicao):
    if not verdade:
        return {"n": 0}
    p, r, f1, _ = precision_recall_fscore_support(
        verdade, predicao, labels=["fresca", "deteriorada"], zero_division=0)
    return {"n": len(verdade), "acuracia": float(accuracy_score(verdade, predicao)),
            "precisao_deteriorada": float(p[1]), "revocacao_deteriorada": float(r[1]),
            "f1_deteriorada": float(f1[1]), "f1_macro": float(np.mean(f1))}


def _melhor_limiar(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    if not len(scores):
        raise ValueError("Nenhuma amostra válida para calibrar o limiar clássico.")
    candidatos = np.unique(np.concatenate(([-1e-6, 0], np.quantile(scores, np.linspace(0, 1, 301)), [1.])))
    melhores = []
    maior = -1.0
    for t in candidatos:
        preds = np.where(scores > t, "deteriorada", "fresca")
        valor = f1_score(labels, preds, labels=["fresca", "deteriorada"], average="macro", zero_division=0)
        if valor > maior + 1e-10:
            maior, melhores = valor, [float(t)]
        elif abs(valor - maior) <= 1e-10:
            melhores.append(float(t))
    return float(np.median(melhores)), float(maior)


def calibrar(dados: str = "dados", saida: str = "modelos/limiares_classico.json", limite: int = 0):
    frame, base = carregar_manifesto(dados)
    val = frame[frame.divisao == "validacao"].copy()
    if limite:
        val = val.groupby("classe", group_keys=False).head(limite)
    coletadas = []
    erros = []
    for i, linha in enumerate(val.itertuples(index=False), 1):
        try:
            res = analisar_classico(abrir_rgb(base / linha.arquivo), linha.especie)
            coletadas.append({"especie": linha.especie, "estado": linha.estado,
                              "score": res.pontuacao, "arquivo": linha.arquivo})
        except (ValueError, OSError) as e:
            erros.append({"arquivo": linha.arquivo, "erro": str(e)})
        if i % 200 == 0:
            print(f"Calibração: {i}/{len(val)} imagens", flush=True)
    tabela = pd.DataFrame(coletadas)
    if tabela.empty:
        raise RuntimeError("Nenhuma imagem foi processada com sucesso.")
    limiares, metricas = {}, {}
    for especie in ESPECIES:
        sub = tabela[tabela.especie == especie]
        if len(sub) < 2:
            limiares[especie] = LIMIARES_PADRAO[especie]
            metricas[especie] = None
        else:
            t, m = _melhor_limiar(sub.score, sub.estado)
            limiares[especie], metricas[especie] = t, m
        print(f"{especie}: limiar {limiares[especie]:.4f}, F1 macro de validacao {metricas[especie]}")
    saida = Path(saida)
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps({"limiares": limiares, "f1_validacao": metricas,
                                 "amostras": len(tabela), "falhas_segmentacao": len(erros),
                                 "metodo": "Limiares maximizando F1-macro apenas na validação."},
                                indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Calibração salva em {saida} (falhas de segmentação: {len(erros)})")
    return saida


def _salvar_matriz(y_true, y_pred, etiquetas, destino: Path, titulo: str):
    matriz = confusion_matrix(y_true, y_pred, labels=etiquetas)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(matriz, cmap="Blues")
    ax.set_xticks(range(len(etiquetas)), etiquetas, rotation=40, ha="right")
    ax.set_yticks(range(len(etiquetas)), etiquetas)
    ax.set_ylabel("Real")
    ax.set_xlabel("Previsto")
    ax.set_title(titulo)
    for (i, j), value in np.ndenumerate(matriz):
        ax.text(j, i, str(value), ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)


def avaliar(dados: str = "dados", pesos: list[str] | None = None,
            limiares: str = "modelos/limiares_classico.json", saida: str = "resultados",
            lote: int = 32, dispositivo: str = "auto", workers: int = 0,
            limite: int = 0):
    frame, base = carregar_manifesto(dados)
    test = frame[frame.divisao == "teste"].reset_index(drop=True)
    if limite > 0:
        test = test.groupby("classe", group_keys=False).head(limite).reset_index(drop=True)
    if test.empty:
        raise ValueError("Manifesto sem imagens de teste.")
    destino = Path(saida)
    destino.mkdir(parents=True, exist_ok=True)
    threshold = carregar_limiares(limiares)
    linhas = []
    for indice, row in enumerate(test.itertuples(index=False), 1):
        try:
            resultado = analisar_classico(abrir_rgb(base / row.arquivo), row.especie, threshold[row.especie])
            pred = resultado.estado
            score = resultado.pontuacao
            falha = ""
        except (ValueError, OSError) as e:
            pred, score, falha = "indeterminado", np.nan, str(e)
        linhas.append({"arquivo": row.arquivo, "especie_real": row.especie,
                       "estado_real": row.estado, "classe_real": row.classe,
                       "estado_previsto": pred, "pontuacao": score, "erro": falha})
        if indice % 250 == 0:
            print(f"Método clássico: {indice}/{len(test)}", flush=True)
    previsoes_classicas = pd.DataFrame(linhas)
    previsoes_classicas.to_csv(destino / "previsoes_classico.csv", index=False)
    validas = previsoes_classicas[previsoes_classicas.estado_previsto != "indeterminado"]
    resumo = {"amostras_teste": len(test), "classico": {
        "cobertura": len(validas) / len(test),
        "geral": _metricas(validas.estado_real.tolist(), validas.estado_previsto.tolist()),
        "por_especie": {e: _metricas(g.estado_real.tolist(), g.estado_previsto.tolist())
                        for e in ESPECIES for g in [validas[validas.especie_real == e]]},
        "nota": "Métricas excluem falhas de segmentação; ver cobertura e CSV de erros."}}
    if not validas.empty:
        _salvar_matriz(validas.estado_real, validas.estado_previsto,
                      ["fresca", "deteriorada"], destino / "confusao_classico.png", "Método clássico")
    print("Clássico:", resumo["classico"]["geral"])

    for arquivo in (pesos or []):
        from src.models.resnet_classifier import carregar_modelo, _lote
        import torch
        ckpt_path = Path(arquivo)
        if not ckpt_path.exists():
            print(f"Modelo ignorado (arquivo ausente): {ckpt_path}")
            continue
        modelo, dev, meta = carregar_modelo(ckpt_path, dispositivo=dispositivo)
        loader = _lote(lote, test, base, False, workers, dev)
        predicoes, conf = [], []
        with torch.inference_mode():
            for imgs, _ in loader:
                imgs = imgs.to(dev)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    saidas = modelo(imgs)
                probabilidades = saidas.float().softmax(1).cpu().numpy()
                predicoes.extend(probabilidades.argmax(axis=1).tolist())
                conf.extend(probabilidades.max(axis=1).tolist())
        classes_pred = [CLASSES[idx] for idx in predicoes]
        estados_pred = [decompor_classe(c)[1] for c in classes_pred]
        especie_pred = [decompor_classe(c)[0] for c in classes_pred]
        nome = ckpt_path.stem
        pd.DataFrame({"arquivo": test.arquivo, "especie_real": test.especie,
                      "estado_real": test.estado, "classe_real": test.classe,
                      "especie_prevista": especie_pred, "estado_previsto": estados_pred,
                      "classe_prevista": classes_pred, "confianca_classe": conf}).to_csv(
                          destino / f"previsoes_{nome}.csv", index=False)
        resumo[nome] = {
            "condicao_geral": _metricas(test.estado.tolist(), estados_pred),
            "condicao_por_especie": {e: _metricas(test[test.especie == e].estado.tolist(),
                                                   [estados_pred[i] for i in np.where(test.especie.values == e)[0]])
                                    for e in ESPECIES},
            "acuracia_seis_classes": float(accuracy_score(test.classe.tolist(), classes_pred)),
            "f1_macro_seis_classes": float(f1_score(test.classe.tolist(), classes_pred, labels=list(CLASSES), average="macro", zero_division=0)),
            "acuracia_especie": float(accuracy_score(test.especie.tolist(), especie_pred)),
            "arquitetura": meta["arquitetura"], "epoca_checkpoint": meta.get("epoca")}
        _salvar_matriz(test.estado.tolist(), estados_pred, ["fresca", "deteriorada"],
                      destino / f"confusao_{nome}.png", nome)
        _salvar_matriz(test.classe.tolist(), classes_pred, list(CLASSES),
                      destino / f"confusao_6classes_{nome}.png", nome + " (6 classes)")
        print(f"{nome}:", resumo[nome]["condicao_geral"])
    (destino / "resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Relatórios e gráficos em {destino}")
    return resumo
