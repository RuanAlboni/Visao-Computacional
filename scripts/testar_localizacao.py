
"""Executa localizacao e exporta as duas fontes sem misturar caixas."""
import argparse
import json
import io
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image,ImageOps
from src.modules.tomato.semantico import DetectorTomate
from src.modules.tomato.localizacao import executar
from src.modules.tomato.vis_localizacao import desenhar

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("imagem",type=Path);p.add_argument("--pesos",type=Path)
    p.add_argument("--device",default="cpu");p.add_argument("--conf",type=float,default=0.35)
    p.add_argument("--sem-classico",action="store_true")
    args=p.parse_args()
    with Image.open(args.imagem) as im: rgb=np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    detector=DetectorTomate.carregar(args.pesos,args.device) if args.pesos else None
    meta=({"caminho":detector.caminho,"sha256":detector.sha256,"classes":detector.nomes,
           "classes_filtradas":detector.ids} if detector else {})
    dados=executar(rgb,detector,not args.sem_classico,args.conf,metadados_modelo=meta)
    pasta=Path("output/localizacao")/dados["id"];pasta.mkdir(parents=True)
    (pasta/"analise.json").write_text(json.dumps(dados,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    for fonte in ("candidatos_classicos","deteccoes_do_modelo"):
        desenhar(rgb,dados,fonte).save(pasta/(fonte+".png"))
    print("\n".join(dados["logs"]))
    print("Saidas:",pasta.resolve())
