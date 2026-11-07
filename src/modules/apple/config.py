"""Configuração do módulo Apple — substitui constantes.py do FruitVision."""

# Classes do dataset Kaggle (apenas maçã para nosso escopo)
APPLE_CLASSES = (
    "apple_fresh",
    "apple_rotten",
)

# Mapeamento de pastas do dataset Kaggle
FOLDER_ALIAS = {
    "freshapples": "apple_fresh",
    "freshapple": "apple_fresh",
    "rottenapples": "apple_rotten",
    "rottenapple": "apple_rotten",
}

# Mapeamento para nosso schema unificado
CONDITION_MAP = {
    "apple_fresh": "fresh",
    "apple_rotten": "rotten",
}

def folder_to_class(folder_name: str) -> str | None:
    """Converte nome de pasta do Kaggle em classe."""
    import re
    chave = re.sub(r"[^a-z]", "", folder_name.lower())
    return FOLDER_ALIAS.get(chave)

def class_to_condition(classe: str) -> str:
    """Converte classe em condition do schema unificado."""
    return CONDITION_MAP.get(classe, "unknown")

# ============================================================
# Aliases de compatibilidade com o FruitVision original
# (kaggle_loader.py e resnet_classifier.py esperam esses nomes)
# ============================================================

# CLASSES: mesmo formato do constantes.py original (6 classes)
# Mantemos as 6 para compatibilidade com o dataset Kaggle completo
CLASSES = (
    "maca_fresca", "maca_deteriorada",
    "banana_fresca", "banana_deteriorada",
    "laranja_fresca", "laranja_deteriorada",
)

ESPECIES = ("maca", "banana", "laranja")
ESTADOS = ("fresca", "deteriorada")

def classe_da_pasta(nome: str) -> str | None:
    """Converte nome de pasta do Kaggle em classe (formato original)."""
    _ALIAS = {
        "freshapples": "maca_fresca", "freshapple": "maca_fresca",
        "rottenapples": "maca_deteriorada", "rottenapple": "maca_deteriorada",
        "freshbanana": "banana_fresca", "freshbananas": "banana_fresca",
        "rottenbanana": "banana_deteriorada", "rottenbananas": "banana_deteriorada",
        "freshoranges": "laranja_fresca", "freshorange": "laranja_fresca",
        "rottenoranges": "laranja_deteriorada", "rottenorange": "laranja_deteriorada",
    }
    import re
    chave = re.sub(r"[^a-z]", "", nome.lower())
    return _ALIAS.get(chave)

def decompor_classe(classe: str) -> tuple[str, str]:
    """Decompõe 'maca_fresca' em ('maca', 'fresca') — formato original."""
    especie, estado = classe.split("_", maxsplit=1)
    if especie not in ESPECIES or estado not in ESTADOS:
        raise ValueError(f"Classe não suportada: {classe}")
    return especie, estado
