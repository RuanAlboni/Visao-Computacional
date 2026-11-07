"""Inspeciona somente metadados de checkpoint de origem confiavel."""
import sys
from pathlib import Path
import torch
if __name__=="__main__":
    if len(sys.argv)!=2:raise SystemExit("Uso: python scripts/inspecionar_checkpoint_alimentos.py pesos.pth")
    p=Path(sys.argv[1]);ck=torch.load(p,map_location="cpu",weights_only=True)
    if not isinstance(ck,dict):raise ValueError("Checkpoint nao e dicionario")
    print("Chaves:",list(ck))
    for chave in ("arquitetura","classes","tamanho","epoca","f1_validacao"):
        print(chave,":",ck.get(chave,"ausente"))
