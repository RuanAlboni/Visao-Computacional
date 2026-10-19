"""CLI: preparar, calibrar, treinar, avaliar."""
from __future__ import annotations
import argparse


def executar():
    parser = argparse.ArgumentParser(description="FruitVision — visão clássica + aprendizado profundo")
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("preparar", help="Cria manifesto sem duplicar o dataset")
    p.add_argument("--origem", required=True, help="Pasta extraída do Kaggle (possui train/test)")
    p.add_argument("--saida", default="dados")
    p.add_argument("--validacao", type=float, default=0.20)
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--limite-por-classe", type=int, default=0, help="Teste rápido, 0 = tudo")

    p = sub.add_parser("calibrar", help="Ajusta limiares clássicos na validação")
    p.add_argument("--dados", default="dados")
    p.add_argument("--saida", default="modelos/limiares_classico.json")
    p.add_argument("--limite", type=int, default=0, help="Imagens por classe na validação, 0 = todas")

    p = sub.add_parser("treinar", help="Treina CNN do zero ou ResNet-18")
    p.add_argument("--dados", default="dados")
    p.add_argument("--arquitetura", choices=["cnn", "resnet18"], default="resnet18")
    p.add_argument("--saida", default=None)
    p.add_argument("--epocas", type=int, default=25)
    p.add_argument("--lote", type=int, default=32)
    p.add_argument("--paciencia", type=int, default=6)
    p.add_argument("--taxa", type=float, default=None)
    p.add_argument("--sem-pretreino", action="store_true")
    p.add_argument("--dispositivo", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--workers", type=int, default=0, help="Recomendado iniciar com 0 no Windows")
    p.add_argument("--semente", type=int, default=42)

    p = sub.add_parser("avaliar", help="Avalia a parte clássica e checkpoints da IA no conjunto de teste")
    p.add_argument("--dados", default="dados")
    p.add_argument("--pesos", nargs="*", default=[], help="Um ou mais checkpoints .pth")
    p.add_argument("--limiares", default="modelos/limiares_classico.json")
    p.add_argument("--saida", default="resultados")
    p.add_argument("--lote", type=int, default=32)
    p.add_argument("--limite", type=int, default=0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--dispositivo", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.comando == "preparar":
        from src.dados import preparar
        preparar(args.origem, args.saida, args.validacao, args.semente, args.limite_por_classe)
    elif args.comando == "calibrar":
        from src.avaliacao import calibrar
        calibrar(args.dados, args.saida, args.limite)
    elif args.comando == "treinar":
        from src.profundo import treinar
        destino = args.saida or f"modelos/{args.arquitetura}.pth"
        treinar(dados=args.dados, arquitetura=args.arquitetura, saida=destino,
               epocas=args.epocas, lote=args.lote, paciencia=args.paciencia, taxa=args.taxa,
               sem_pretreino=args.sem_pretreino, dispositivo=args.dispositivo,
               workers=args.workers, semente=args.semente)
    elif args.comando == "avaliar":
        from src.avaliacao import avaliar
        avaliar(args.dados, args.pesos, args.limiares, args.saida, args.lote,
                args.dispositivo, args.workers, args.limite)


if __name__ == "__main__":
    executar()
