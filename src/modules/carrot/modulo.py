"""Adaptador cenoura; usa detector compartilhado, sem inventar estado."""
from src.modules.base import BaseFruitModule
from src.schemas.fruit import FruitResult,FruitDetection,FruitCondition,BoundingBox

class CarrotModule(BaseFruitModule):
    fruit_type="carrot"
    def __init__(self,pesos_detector="models/yolo11n.pt",pesos_estado=None,device="cpu"):
        self.config=(pesos_detector,pesos_estado,device);self.motor=None
    def load_model(self):
        from src.alimentos_v2.fluxo import Motor
        self.motor=Motor(*self.config)
    def detect(self,image_path):
        import numpy as np
        from PIL import Image,ImageOps
        if self.motor is None:self.load_model()
        with Image.open(image_path) as im:rgb=np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        dados=self.motor.analisar(rgb,produtos=["cenoura"]);deteccoes=[]
        for d in dados["deteccoes"]:
            x1,y1,x2,y2=d["bbox"]
            cond={"fresco":FruitCondition.FRESH,"deteriorado":FruitCondition.ROTTEN}.get(d["estado"],FruitCondition.UNKNOWN)
            deteccoes.append(FruitDetection("carrot",cond,float(d.get("score",0)),d["detection_score"],
                BoundingBox(x1,y1,x2-x1,y2-y1),"yolo+multiclasse" if "score" in d else "yolo_estado_nao_avaliado"))
        return FruitResult(str(image_path),deteccoes)
    def detect_batch(self,image_paths):return [self.detect(p) for p in image_paths]
