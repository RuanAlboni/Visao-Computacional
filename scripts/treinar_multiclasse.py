"""CLI do treinamento multiclasse com suporte a pastas e manifesto CSV."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description="Treina ResNet-18/CNN com validação e early stopping.")
    p.add_argument("--dados", required=True, type=Path, help="Raiz com train/ e val/ por classe.")
    p.add_argument("--saida", default="models/novo_multiclasse.pth", type=Path)
    p.add_argument("--arquitetura", choices=["resnet18", "cnn"], default="resnet18")
    p.add_argument("--epocas", type=int, default=25)
    p.add_argument("--lote", type=int, default=32)
    p.add_argument("--paciencia", type=int, default=6)
    p.add_argument("--taxa", type=float, default=None)
    p.add_argument("--device", "--dispositivo", dest="dispositivo", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--sem-pretreino", action="store_true")
    a = p.parse_args()

    from src.alimentos_v2.treinamento import treinar

    treinar(dados=a.dados, saida=a.saida, arquitetura=a.arquitetura,
           epocas=a.epocas, lote=a.lote, paciencia=a.paciencia,
           taxa=a.taxa, sem_pretreino=a.sem_pretreino,
           dispositivo=a.dispositivo, workers=a.workers, semente=a.semente)


if __name__ == "__main__":
    main()
