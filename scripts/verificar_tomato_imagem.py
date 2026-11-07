"""Executar na raiz: python scripts/verificar_tomato_imagem.py imagem.jpg."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.modules.tomato.modulo import carregar_imagem
from src.pipeline import analisar
from src.utils.anotar import salvar

def main():
    if len(sys.argv)!=2:
        raise SystemExit("Uso: python scripts/verificar_tomato_imagem.py caminho_da_imagem")
    imagem=carregar_imagem(sys.argv[1])
    analise=analisar("tomate",imagem,armazenamento="bancada")
    pasta=Path("output/revisao_tomato"); pasta.mkdir(parents=True,exist_ok=True)
    salvar(imagem,analise.resultados,pasta/"candidatos.png")
    (pasta/"analise.json").write_text(json.dumps(analise.para_dict(),ensure_ascii=False,indent=2),encoding="utf-8")
    for linha in analise.logs(): print(linha)
    print("Artefatos:",pasta.resolve())

if __name__=="__main__": main()
