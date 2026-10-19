"""Importacao sem copiar as imagens do dataset; separacao estratificada por classe."""
from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .constantes import CLASSES, classe_da_pasta, decompor_classe

EXTENSOES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _subpastas_classes(pasta: Path) -> dict[str, Path]:
    if not pasta.is_dir():
        return {}
    resultado = {}
    for p in pasta.iterdir():
        if p.is_dir():
            classe = classe_da_pasta(p.name)
            if classe is not None:
                resultado[classe] = p
    return resultado


def encontrar_treino_teste(origem: Path) -> tuple[Path, Path]:
    origem = origem.resolve()
    if not origem.exists():
        raise FileNotFoundError(f"Diretorio nao existe: {origem}")
    candidatas = [p for p in origem.rglob("*") if p.is_dir() and p.name.lower() == "train"]
    if origem.name.lower() == "train":
        candidatas.insert(0, origem)
    for pasta_train in candidatas:
        colegas = {p.name.lower(): p for p in pasta_train.parent.iterdir() if p.is_dir()}
        pasta_test = colegas.get("test")
        if pasta_test and len(_subpastas_classes(pasta_train)) == 6 and len(_subpastas_classes(pasta_test)) == 6:
            return pasta_train, pasta_test
    raise ValueError("Estrutura não encontrada. Esperado: .../train/{6 classes}/ e .../test/{6 classes}/")


def _base_foto(nome: str) -> str:
    """Heuristica de agrupamento por nome; NAO garante detectar todas as augmentations."""
    texto = Path(nome).stem.lower()
    padroes = (
        r"^(?:rotated[_-]?by[_-]?\d+[_-]?)",
        r"^(?:rotated|rotation|rotate|flipped|flip|horizontal[_-]?flip|vertical[_-]?flip)[_-]?\d*[_-]?",
        r"(?:[_-](?:aug|augmentation|copy|flip|rotate|rotated)[_-]?\d*)$",
    )
    anterior = None
    while anterior != texto:
        anterior = texto
        for regex in padroes:
            texto = re.sub(regex, "", texto)
    return texto or Path(nome).stem.lower()


def _enumerar(pasta: Path, divisao: str, base: Path) -> list[dict]:
    linhas = []
    for classe, sub in sorted(_subpastas_classes(pasta).items()):
        especie, estado = decompor_classe(classe)
        for arquivo in sorted(sub.rglob("*")):
            if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES:
                linhas.append({"arquivo": arquivo.resolve().relative_to(base).as_posix(),
                               "especie": especie, "estado": estado, "classe": classe,
                               "divisao": divisao, "grupo": _base_foto(arquivo.name)})
    return linhas


def preparar(origem: str | Path, saida: str | Path = "dados", validacao: float = 0.2,
             semente: int = 42, limite_por_classe: int = 0) -> pd.DataFrame:
    if not 0.05 <= validacao <= 0.5:
        raise ValueError("Validação deve estar entre 0.05 e 0.5.")
    origem = Path(origem).resolve()
    treino_dir, teste_dir = encontrar_treino_teste(origem)
    linhas = _enumerar(treino_dir, "treino", origem) + _enumerar(teste_dir, "teste", origem)
    frame = pd.DataFrame(linhas)
    if frame.empty:
        raise ValueError("Nenhuma imagem reconhecida.")
    if set(frame.classe.unique()) != set(CLASSES):
        raise ValueError("Devem existir imagens para as 6 classes.")
    if limite_por_classe > 0:
        amostras = []
        for _, sub in frame.groupby(["divisao", "classe"]):
            amostras.append(sub.sample(n=min(len(sub), limite_por_classe), random_state=semente))
        frame = pd.concat(amostras, ignore_index=True)
    partes = []
    for classe in CLASSES:
        sub = frame[(frame.classe == classe) & (frame.divisao == "treino")].copy()
        if sub.grupo.nunique() < 2:
            raise ValueError(f"Poucas fotos independentes na classe {classe}.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=validacao, random_state=semente)
        indices_treino, indices_validacao = next(splitter.split(sub, groups=sub.grupo))
        parte_treino = sub.iloc[indices_treino].copy()
        parte_val = sub.iloc[indices_validacao].copy()
        parte_val.loc[:, "divisao"] = "validacao"
        partes.extend([parte_treino, parte_val])
    partes.append(frame[frame.divisao == "teste"].copy())
    final = pd.concat(partes, ignore_index=True)
    final = final.sort_values(["divisao", "classe", "arquivo"]).reset_index(drop=True)
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    final.to_csv(saida / "manifesto.csv", index=False, encoding="utf-8")
    meta = {"origem": str(origem), "dataset": "Kaggle sriramr/fruits-fresh-and-rotten-for-classification",
            "semente": semente, "fracao_validacao": validacao,
            "classes": list(CLASSES), "contagens": {k: int(v) for k, v in final.groupby("divisao").size().items()}}
    (saida / "metadados.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    # Um grupo com o mesmo nome em treino/teste nao prova ser igual, mas merece inspecao.
    conjunto_dev = set(final[final.divisao != "teste"].apply(lambda x: (x.classe, x.grupo), axis=1))
    conjunto_test = set(final[final.divisao == "teste"].apply(lambda x: (x.classe, x.grupo), axis=1))
    suspeitos = len(conjunto_dev & conjunto_test)
    print("Imagens por divisão e classe:\n", pd.crosstab(final.classe, final.divisao).to_string())
    print(f"Grupos de nomes possivelmente compartilhados entre desenvolvimento e teste: {suspeitos}.")
    print("ATENÇÃO: imagens aumentadas visualmente semelhantes podem ter nomes diferentes. Audite as duplicatas antes de publicar métricas.")
    print(f"Manifesto salvo em: {saida / 'manifesto.csv'}")
    return final


def carregar_manifesto(dados: str | Path = "dados") -> tuple[pd.DataFrame, Path]:
    dados = Path(dados)
    manifest = dados / "manifesto.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Execute primeiro 'python main.py preparar ...'. Arquivo ausente: {manifest}")
    frame = pd.read_csv(manifest, dtype=str)
    meta = json.loads((dados / "metadados.json").read_text(encoding="utf-8"))
    base = Path(meta["origem"])
    if not base.exists():
        raise FileNotFoundError(f"Dataset movido: {base}. Reexecute a preparação com o novo caminho.")
    return frame, base
