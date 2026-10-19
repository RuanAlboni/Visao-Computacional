"""Nomes e mapeamento das seis classes do dataset Kaggle."""
from __future__ import annotations
import re

ESPECIES = ("maca", "banana", "laranja")
ESTADOS = ("fresca", "deteriorada")
CLASSES = tuple(f"{especie}_{estado}" for especie in ESPECIES for estado in ESTADOS)
ROTULOS_ESPECIES = {"maca": "Maçã", "banana": "Banana", "laranja": "Laranja"}
ROTULOS_ESTADOS = {"fresca": "Fresca", "deteriorada": "Deteriorada"}

_ALIAS = {
    "freshapples": "maca_fresca", "freshapple": "maca_fresca",
    "rottenapples": "maca_deteriorada", "rottenapple": "maca_deteriorada",
    "freshbanana": "banana_fresca", "freshbananas": "banana_fresca",
    "rottenbanana": "banana_deteriorada", "rottenbananas": "banana_deteriorada",
    "freshoranges": "laranja_fresca", "freshorange": "laranja_fresca",
    "rottenoranges": "laranja_deteriorada", "rottenorange": "laranja_deteriorada",
}

def classe_da_pasta(nome: str) -> str | None:
    chave = re.sub(r"[^a-z]", "", nome.lower())
    return _ALIAS.get(chave)


def decompor_classe(classe: str) -> tuple[str, str]:
    especie, estado = classe.split("_", maxsplit=1)
    if especie not in ESPECIES or estado not in ESTADOS:
        raise ValueError(f"Classe nao suportada: {classe}")
    return especie, estado


def rotulo(classe: str) -> str:
    especie, estado = decompor_classe(classe)
    return f"{ROTULOS_ESPECIES[especie]} · {ROTULOS_ESTADOS[estado]}"
