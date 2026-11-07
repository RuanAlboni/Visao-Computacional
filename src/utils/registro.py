"""Persistencia dos logs da aplicacao em arquivo, alem dos logs na tela."""

import logging
from datetime import datetime
from pathlib import Path


def configurar(pasta="logs", nome="smart_fridge_vision"):
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    arquivo = pasta / f"{datetime.now():%Y%m%d}.log"

    logger = logging.getLogger(nome)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        manipulador = logging.FileHandler(arquivo, encoding="utf-8")
        manipulador.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(manipulador)
    return logger


def gravar(logger, analise):
    logger.info(
        "modulo=%s frutas=%s apropriados=%s inapropriados=%s tempo=%.2fs",
        analise.modulo, analise.resumo.get("total"),
        analise.resumo.get("apropriados"), analise.resumo.get("inapropriados"),
        analise.duracao_s,
    )
    for linha in analise.logs():
        logger.info(linha)