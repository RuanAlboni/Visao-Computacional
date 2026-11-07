"""Bloco 1: candidatos classicos + triagem visual, sem declaracoes de consumo."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from src.modules.base import BaseFruitModule
from src.modules.tomato.deteccao import propor_candidatos, validar_rgb
from src.classic.tomato_rules import analisar_tomate
from src.schemas.fruit import BoundingBox, FruitCondition, FruitDetection, FruitResult

@dataclass
class ResultadoTomate:
    classe: str
    estado: str
    mensagem: str
    bbox: tuple
    metricas: dict = field(default_factory=dict)
    veredito: str = "inconclusivo"
    apropriado: bool | None = None
    confianca: float | None = None
    prob_fresco: float | None = None
    dias_restantes: int | None = None
    recomendacao: str = "Identidade e seguranca alimentar nao confirmadas. Prazo nao estimado."
    fonte: str = "classico_experimental"
    incerto: bool = True
    detalhes: dict = field(default_factory=dict)

    def para_dict(self):
        return asdict(self)

    def __str__(self):
        return f"{self.classe} | {self.mensagem} | prazo nao estimado"

def carregar_imagem(caminho):
    with Image.open(caminho) as imagem:
        return np.asarray(ImageOps.exif_transpose(imagem).convert("RGB"))

class TomatoModule(BaseFruitModule):
    fruit_type = "tomato"

    def __init__(self, usar_classico=True, usar_profundo=False, modelo_path=None,
                 armazenamento="geladeira", area_min=120):
        if usar_profundo:
            raise RuntimeError("Ramo profundo indisponivel neste bloco; requer classificador e pesos de tomate validados.")
        if not usar_classico:
            raise ValueError("Nenhuma abordagem habilitada.")
        self.armazenamento = armazenamento
        self.area_min = area_min
        self.logs = []

    def load_model(self) -> None:
        raise RuntimeError("Nenhum modelo profundo integrado no bloco 1.")

    def analisar(self, imagem_rgb, armazenamento=None):
        rgb = validar_rgb(imagem_rgb)
        candidatos, self.logs = propor_candidatos(rgb, area_min=self.area_min)
        resultados = []
        for indice, candidato in enumerate(candidatos, 1):
            x1,y1,x2,y2 = candidato.bbox
            saida = analisar_tomate(rgb[y1:y2,x1:x2], mascara_objeto=candidato.mascara)
            resultados.append(ResultadoTomate(
                classe=saida["classe"], estado=saida["estado"], mensagem=saida["motivo"],
                bbox=candidato.bbox, metricas=saida["metricas"],
                detalhes={"id": indice, "deteccao": candidato.metodo,
                          "identidade_confirmada": False,
                          "armazenamento": armazenamento or self.armazenamento}))
        return resultados

    def para_fruit_result(self, resultados, image_path=""):
        deteccoes = []
        for r in resultados:
            x1,y1,x2,y2 = r.bbox
            # 0.0 significa confianca nao estimada no schema legado (campo float).
            deteccoes.append(FruitDetection(
                fruit_type=self.fruit_type, condition=FruitCondition.UNKNOWN,
                confidence=0.0, detection_confidence=0.0,
                bbox=BoundingBox(x=x1,y=y1,width=x2-x1,height=y2-y1),
                method="candidato_classico_nao_confirmado"))
        return FruitResult(image_path=str(image_path), detections=deteccoes)

    def detect(self, image_path) -> FruitResult:
        rgb = carregar_imagem(image_path) if isinstance(image_path, (str,Path)) else image_path
        resultados = self.analisar(rgb)
        nome = str(image_path) if isinstance(image_path,(str,Path)) else ""
        return self.para_fruit_result(resultados, nome)

    def detectar(self, imagem_rgb):
        return self.detect(imagem_rgb)

    def detect_batch(self, image_paths: list) -> list:
        return [self.detect(caminho) for caminho in image_paths]

    @staticmethod
    def resumo(resultados):
        return {"total": len(resultados), "candidatos": len(resultados),
                "confirmados": 0, "apropriados": 0, "inapropriados": 0,
                "incertos": len(resultados), "dias_medios_restantes": None,
                "proximo_vencimento": None,
                "sem_alteracoes_evidentes": sum(r.estado == "sem_alteracoes_visuais_evidentes" for r in resultados)}
