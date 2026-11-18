"""Encontra limiares clássicos em data/alimentos/val/ e gera JSON compatível com o painel."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description="Calibra limiares clássicos por espécie no conjunto val/.")
    p.add_argument("--dados", required=True, type=Path, help="Raiz com val/<classe>/*.jpg.")
    p.add_argument("--saida", type=Path, default=Path("models/limiares_classico_novos.json"))
    p.add_argument("--limite", type=int, default=0, help="Máximo de imagens por classe (0 = todas).")
    p.add_argument("--csv", type=Path, default=None, help="Arquivo CSV opcional com scores por imagem.")
    args = p.parse_args()

    from src.alimentos_v2.calibracao import calibrar
    calibrar(args.dados, args.saida, args.limite, args.csv)


if __name__ == "__main__":
    main()
