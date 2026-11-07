"""CLI: python -m src.modules.tomato.cli --imagem tomate.jpg --salvar-png saida.png"""

import argparse
import json
from pathlib import Path

from src.modules.tomato.modulo import TomatoModule, carregar_imagem


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analise de tomates: consumo e dias restantes")
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--imagem")
    grupo.add_argument("--pasta")
    parser.add_argument("--modelo", default=None)
    parser.add_argument("--yolo", default=None)
    parser.add_argument("--armazenamento", default="geladeira", choices=["geladeira", "bancada"])
    parser.add_argument("--limiar", type=float, default=0.50)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--salvar-png", default=None)
    args = parser.parse_args(argv)

    detector_yolo = None
    if args.yolo and Path(args.yolo).exists():
        from ultralytics import YOLO
        detector_yolo = YOLO(args.yolo)

    modulo = TomatoModule(modelo_path=args.modelo, detector_yolo=detector_yolo,
                          armazenamento=args.armazenamento, limiar=args.limiar)

    if args.pasta:
        resultados = modulo.analisar_pasta(args.pasta, args.armazenamento)
        if args.json:
            print(json.dumps([r.para_dict() for r in resultados], ensure_ascii=False, indent=2))
        else:
            for r in resultados:
                print(f"{Path(r.detalhes.get('arquivo', '?')).name}: {r}")
        print("\nRESUMO:", json.dumps(modulo.resumo(resultados), ensure_ascii=False))
        return 0

    imagem = carregar_imagem(args.imagem)
    if imagem is None:
        raise SystemExit(f"Nao foi possivel abrir {args.imagem}")
    resultados = modulo.analisar(imagem)
    if args.json:
        print(json.dumps([r.para_dict() for r in resultados], ensure_ascii=False, indent=2))
    else:
        print(f"{len(resultados)} tomate(s) encontrado(s)\n")
        for i, r in enumerate(resultados, start=1):
            print(f"Tomate {i}: {r}")
            print(f"   {r.recomendacao}")
            if r.incerto:
                print("   Aviso: decisao proxima do limiar, revise manualmente.")
        print(f"\nRESUMO: {json.dumps(modulo.resumo(resultados), ensure_ascii=False)}")
    if args.salvar_png:
        from src.modules.tomato.visualizar import salvar_anotado
        salvar_anotado(imagem, resultados, args.salvar_png, titulo=f"Tomate ({args.armazenamento})")
        print(f"Imagem anotada salva em {args.salvar_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())