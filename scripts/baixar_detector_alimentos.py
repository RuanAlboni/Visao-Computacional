"""Download explicito de detector COCO oficial. Requer internet; verificar licenca Ultralytics."""
from pathlib import Path
import shutil
from ultralytics import YOLO
if __name__=="__main__":
    modelo=YOLO("yolo11n.pt")
    mapa=modelo.names
    exigidos={"carrot","broccoli","banana","apple","orange"}
    if not exigidos.issubset(set(mapa.values())):raise RuntimeError("Classes inesperadas")
    origem=Path(modelo.ckpt_path).resolve();destino=Path("models/yolo11n.pt").resolve()
    destino.parent.mkdir(parents=True,exist_ok=True)
    if origem!=destino:shutil.copy2(origem,destino)
    print("Pesos:",destino)
    print("Classes:",{k:v for k,v in mapa.items() if v in exigidos})
