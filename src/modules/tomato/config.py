"""Configuracao do modulo tomate (deteccao + classificacao + veredito de consumo)."""

ESPECIE = "tomate"

# Ordem oficial das classes: indice 0 = fresco, indice 1 = deteriorado.
CLASSES = ("tomate_fresco", "tomate_deteriorado")
CLASSE_FRESCO, CLASSE_DETERIORADO = CLASSES

ESTADOS = {"fresco": "apropriado", "deteriorado": "inapropriado"}

LIMIAR_APROPRIADO = 0.50          # prob(fresco) >= limiar -> apropriado
MARGEM_INCERTEZA = 0.08           # faixa em torno do limiar considerada incerta

# Vida util maxima em dias (do estagio firme-maduro), por armazenamento.
# 8-10 dias em temperatura otima; ate ~2 semanas refrigerado.
VIDA_UTIL_MAX_DIAS = {"geladeira": 12, "bancada": 5}
ARMAZENAMENTO_PADRAO = "geladeira"

# Fusao das duas abordagens da disciplina
PESO_PROFUNDO = 0.70              # ResNet-18
PESO_CLASSICO = 0.30              # pipeline classico (HSV/regras)

CONF_MIN_DETECCAO = 0.35
AREA_MIN_REGIAO = 1200            # px^2
MAX_REGIOES = 5

# Aceita datasets tipo "fresh_tomatoes" e albuns do PlantVillage (Tomato___Late_blight...)
PASTAS_DETERIORADO = (
    "rotten", "deteriorado", "spoiled", "bad", "rot", "blight", "mold",
    "mould", "spot", "mosaic", "bacterial", "curl", "yellow", "disease",
)
PASTAS_FRESCO = ("fresh", "fresco", "healthy", "saudavel")


def classe_da_pasta(nome_pasta: str) -> str:
    nome = str(nome_pasta).lower()
    if any(chave in nome for chave in PASTAS_DETERIORADO):
        return CLASSE_DETERIORADO
    if any(chave in nome for chave in PASTAS_FRESCO):
        return CLASSE_FRESCO
    raise ValueError(f"Pasta desconhecida para tomate: {nome_pasta}")


def decompor_classe(classe: str) -> tuple:
    """('tomate_fresco') -> ('tomate', 'fresco')."""
    if classe not in CLASSES:
        raise ValueError(f"Classe invalida: {classe}")
    especie, estado = classe.split("_", 1)
    return especie, estado


def veredito_da_classe(classe: str) -> str:
    return ESTADOS[decompor_classe(classe)[1]]