"""Auditoria do modulo tomate: aponta quais arquivos estao inconsistentes."""

import importlib
import inspect
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

ARQUIVOS = [
    "src/modules/base.py", "src/schemas/fruit.py",
    "src/modules/tomato/__init__.py", "src/modules/tomato/config.py",
    "src/modules/tomato/vida_util.py", "src/modules/tomato/deteccao.py",
    "src/modules/tomato/modulo.py", "src/modules/tomato/visualizar.py",
    "src/modules/tomato/cli.py", "src/classic/features.py",
    "src/classic/tomato_rules.py", "src/classic/_classico_original.py",
    "src/models/resnet_classifier.py", "src/data/kaggle_loader.py",
    "src/utils/avaliacao.py", "tests/test_tomato.py", "tests/conftest.py",
]

MARCADORES = {
    "src/modules/tomato/modulo.py": [
        ("import dataclasses", "dataclass do ResultadoClassico reconhecido"),
        ("_para_dicionario", "conversao de dataclass/objeto"),
        ("especie_classico", "especie passada ao classico"),
        ("limiar_classico", "limiar do classico"),
        ("tomato_rules", "pipeline classico do tomate ligado"),
        ('saida.get("confianca"', "confianca explicita respeitada"),
        ("fruit_type", "atributo exigido pelo ABC"),
        ("def detect_batch", "contrato BaseFruitModule"),
        ("def load_model", "contrato BaseFruitModule"),
    ],
    "src/modules/tomato/deteccao.py": [
        ("except Exception:", "segmentar_fruta tolerante a ValueError"),
        ("frame_inteiro", "fallback de deteccao"),
    ],
    "src/classic/tomato_rules.py": [
        ("def analisar_tomate", "classificador classico do tomate"),
    ],
}

IMPORTS = [
    "src.schemas.fruit", "src.modules.base", "src.modules.tomato.config",
    "src.modules.tomato.vida_util", "src.modules.tomato.deteccao",
    "src.modules.tomato.modulo", "src.classic.features",
    "src.classic.tomato_rules", "src.classic._classico_original",
    "src.models.resnet_classifier", "src.data.kaggle_loader",
    "src.utils.avaliacao",
]


def bloco(titulo):
    print("\n" + "=" * 70)
    print(titulo)
    print("=" * 70)


bloco("1) ARQUIVOS: existencia, tamanho e data")
faltando = []
for relativo in ARQUIVOS:
    caminho = RAIZ / relativo
    if not caminho.exists():
        faltando.append(relativo)
        print(f"FALTA   {relativo}")
        continue
    est = caminho.stat()
    print(f"ok      {relativo:<42} {est.st_size:>7} bytes")

bloco("2) MARCADORES: cada correcao ja aplicada?")
for relativo, marcadores in MARCADORES.items():
    caminho = RAIZ / relativo
    if not caminho.exists():
        print(f"{relativo}: ARQUIVO AUSENTE")
        continue
    texto = caminho.read_text()
    print(f"\n{relativo}")
    for marca, descricao in marcadores:
        estado = "OK  " if marca in texto else "FALTA"
        print(f"  [{estado}] {marca:<26} {descricao}")

bloco("3) IMPORTS: o que quebra ao carregar")
for modulo in IMPORTS:
    try:
        importlib.import_module(modulo)
        print(f"ok      {modulo}")
    except Exception as erro:
        print(f"ERRO    {modulo} -> {type(erro).__name__}: {erro}")

bloco("4) ASSINATURAS REAIS")
def mostrar(nome_modulo, atributo):
    try:
        alvo = getattr(importlib.import_module(nome_modulo), atributo)
        print(f"{nome_modulo}.{atributo}: {inspect.signature(alvo)}")
    except Exception as erro:
        print(f"{nome_modulo}.{atributo}: indisponivel ({type(erro).__name__})")

mostrar("src.classic._classico_original", "analisar_classico")
mostrar("src.classic._classico_original", "carregar_limiares")
mostrar("src.classic.features", "segmentar_fruta")
mostrar("src.classic.tomato_rules", "analisar_tomate")
mostrar("src.models.resnet_classifier", "prever_rgb")

try:
    from src.schemas.fruit import FruitResult
    print("FruitResult:", list(getattr(FruitResult, "__dataclass_fields__", {})))
except Exception as erro:
    print("FruitResult indisponivel:", erro)

try:
    from src.modules.base import BaseFruitModule
    print("BaseFruitModule abstratos:", sorted(BaseFruitModule.__abstractmethods__))
except Exception as erro:
    print("BaseFruitModule indisponivel:", erro)

bloco("5) EXECUCAO: analise de uma imagem sintetica")
try:
    import numpy as np
    from src.modules.tomato import TomatoModule

    imagem = np.zeros((120, 160, 3), dtype=np.uint8)
    imagem[20:100, 30:130] = (200, 40, 30)
    modulo = TomatoModule(usar_profundo=False, usar_classico=True, limiar_classico=0.50)
    print("resultado:", modulo.analisar(imagem)[0])
    print("erro classico:", modulo._erro_classico)
except Exception as erro:
    print(f"ERRO na execucao -> {type(erro).__name__}: {erro}")

bloco("RESUMO")
print(f"arquivos ausentes ({len(faltando)}):", faltando or "nenhum")
print("Linhas FALTA/ERRO apontam os arquivos candidatos.")
