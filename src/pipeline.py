"""Registro de modulos e resultado auditavel da aplicacao."""
import time
from dataclasses import dataclass, field
MODULOS = {}

def registrar(nome, fabrica, disponivel=True, descricao=""):
    MODULOS[nome] = {"fabrica": fabrica,"disponivel": disponivel,"descricao": descricao}

def listar():
    return {k:{"disponivel":v["disponivel"],"descricao":v["descricao"]} for k,v in MODULOS.items()}

@dataclass
class Analise:
    modulo: str
    resultados: list
    resumo: dict
    duracao_s: float
    parametros: dict
    mensagens: list = field(default_factory=list)

    def logs(self):
        linhas = [f"modulo={self.modulo} tempo={self.duracao_s:.3f}s", *self.mensagens]
        for i,r in enumerate(self.resultados,1):
            linhas.append(f"#{i} bbox={r.bbox} metodo={r.fonte} estado={r.estado}")
            linhas.append(f"#{i} metricas={r.metricas}")
            linhas.append(f"#{i} {r.mensagem}")
        linhas.append("Consumo inconclusivo; prazo e confianca nao estimados.")
        return linhas

    def para_dict(self):
        return {"modulo":self.modulo,"duracao_s":self.duracao_s,"parametros":self.parametros,
                "resumo":self.resumo,"logs":self.logs(),"resultados":[r.para_dict() for r in self.resultados]}

def analisar(nome_modulo, imagem_rgb, **kwargs):
    entrada = MODULOS[nome_modulo]
    if not entrada["disponivel"]:
        raise RuntimeError("Modulo indisponivel.")
    t = time.perf_counter()
    modulo = entrada["fabrica"](**kwargs)
    resultados = modulo.analisar(imagem_rgb)
    return Analise(nome_modulo,resultados,modulo.resumo(resultados),time.perf_counter()-t,
                   dict(kwargs),list(getattr(modulo,"logs",[])))

def _tomate(**kwargs):
    from src.modules.tomato import TomatoModule
    return TomatoModule(**kwargs)

registrar("tomate",_tomate,descricao="Candidatos classicos experimentais; identidade nao confirmada.")
