"""Verificacoes do modulo tomate sem dependencia de pytest."""

import numpy as np

from src.modules.tomato import CLASSES, TomatoModule, classe_da_pasta, decompor_classe
from src.modules.tomato.vida_util import dias_restantes, mensagem, veredito

falhas = []


def checar(nome, condicao):
    print(("OK  " if condicao else "FALHOU ") + nome)
    if not condicao:
        falhas.append(nome)


checar("classes", CLASSES == ("tomate_fresco", "tomate_deteriorado"))
checar("pasta fresca", classe_da_pasta("Tomato___fresh") == "tomate_fresco")
checar("pasta doente", classe_da_pasta("Tomato___Late_blight") == "tomate_deteriorado")
checar("decompor", decompor_classe("tomate_fresco") == ("tomate", "fresco"))

for prob in (1.0, 0.9, 0.7, 0.51):
    checar(f"apropriado p={prob}", veredito(prob) == "apropriado" and dias_restantes(prob) >= 1)

for prob in (0.49, 0.3, 0.0):
    checar(f"inapropriado p={prob}", veredito(prob) == "inapropriado" and dias_restantes(prob) == 0)

checar("geladeira > bancada", dias_restantes(0.9, "geladeira") > dias_restantes(0.9, "bancada"))
checar("mensagem descarte", "descartar" in mensagem(0.2))

imagem = np.zeros((120, 160, 3), dtype=np.uint8)
imagem[20:100, 30:130] = (200, 40, 30)
modulo = TomatoModule(usar_profundo=False, usar_classico=True)
resultados = modulo.analisar(imagem)
checar("analise classica", bool(resultados) and resultados[0].fonte in ("classico", "fusao"))
checar("resumo", modulo.resumo(resultados)["total"] == len(resultados))

try:
    TomatoModule(usar_profundo=False, usar_classico=False).analisar(imagem)
    checar("sem abordagens falha", False)
except RuntimeError:
    checar("sem abordagens falha", True)

print(f"\n{len(falhas)} falha(s)")
raise SystemExit(1 if falhas else 0)