"""Catalogo comum ao detector, classificador e modulos."""
ALIMENTOS={"cenoura":"carrot","brocolis":"broccoli","banana":"banana","maca":"apple","laranja":"orange"}
REVERSO={v:k for k,v in ALIMENTOS.items()}

def decompor(classe):
    produto,sep,estado=str(classe).rpartition("_")
    if not sep or produto not in ALIMENTOS or estado not in {"fresco","fresca","deteriorado","deteriorada"}:
        raise ValueError(f"Classe nao suportada: {classe}")
    return produto,"fresco" if estado in {"fresco","fresca"} else "deteriorado"

def validar_classes(classes):
    if not classes or len(set(classes))!=len(classes): raise ValueError("Classes vazias ou duplicadas.")
    for classe in classes: decompor(classe)
    return tuple(classes)

def escolher_estado(probabilidades,produto,limiar=0.60):
    if not 0<limiar<=1: raise ValueError("Limiar invalido")
    if not probabilidades: raise ValueError("Probabilidades vazias")
    classe=max(probabilidades,key=probabilidades.get)
    previsto,estado=decompor(classe)
    score=float(probabilidades[classe])
    if previsto!=produto:
        return {"estado":"inconclusivo","motivo":"Detector e classificador discordam sobre o alimento.","classe_prevista":classe,"score":score}
    if score<limiar:
        return {"estado":"inconclusivo","motivo":"Score abaixo do limiar de triagem.","classe_prevista":classe,"score":score}
    return {"estado":estado,"motivo":"Previsao visual do classificador; nao confirma seguranca alimentar.","classe_prevista":classe,"score":score}
