"""Um ResNet-18 para todas as classes alimento+condicao. Pastas train/val/test."""
import sys,json,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--dados",required=True,type=Path)
    p.add_argument("--saida",default="models/alimentos_multiclasse.pth",type=Path)
    p.add_argument("--epocas",type=int,default=15);p.add_argument("--lote",type=int,default=16)
    p.add_argument("--device",default="cpu");p.add_argument("--sem-pretreino",action="store_true")
    a=p.parse_args()
    if a.epocas<1 or a.lote<1:raise ValueError("Epocas/lote devem ser positivos")
    import torch
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader
    from src.alimentos_v2.profundo import criar,transformar
    from src.alimentos_v2.catalogo import validar_classes
    from sklearn.metrics import f1_score
    torch.manual_seed(42)
    treino=ImageFolder(a.dados/"train",transformar(True));valid=ImageFolder(a.dados/"val",transformar(False))
    classes=validar_classes(treino.classes)
    if treino.class_to_idx!=valid.class_to_idx:raise ValueError("Classes/indices de train e val diferentes")
    modelo=criar("resnet18",len(classes),not a.sem_pretreino).to(a.device)
    opt=torch.optim.AdamW(modelo.parameters(),lr=1e-4,weight_decay=1e-4)
    criterio=torch.nn.CrossEntropyLoss();melhor=-1;historico=[]
    a.saida.parent.mkdir(parents=True,exist_ok=True)
    for epoca in range(1,a.epocas+1):
        metricas={}
        for nome,ds,treinando in (("train",treino,True),("val",valid,False)):
            modelo.train(treinando);reais=[];previstos=[];perda_total=0
            for x,y in DataLoader(ds,batch_size=a.lote,shuffle=treinando,num_workers=0):
                x,y=x.to(a.device),y.to(a.device)
                with torch.set_grad_enabled(treinando):
                    logits=modelo(x);perda=criterio(logits,y)
                    if treinando:opt.zero_grad();perda.backward();opt.step()
                perda_total+=perda.item()*len(y)
                reais+=y.cpu().tolist();previstos+=logits.argmax(1).detach().cpu().tolist()
            metricas[nome]={"perda":perda_total/len(ds),"f1_macro":float(f1_score(reais,previstos,labels=list(range(len(classes))),average="macro",zero_division=0))}
        historico.append({"epoca":epoca,**metricas});print(json.dumps(historico[-1]),flush=True)
        if metricas["val"]["f1_macro"]>melhor:
            melhor=metricas["val"]["f1_macro"]
            torch.save({"arquitetura":"resnet18","classes":list(classes),"modelo":modelo.state_dict(),"tamanho":224,
                "epoca":epoca,"f1_validacao":melhor,"historico":historico},a.saida)
    a.saida.with_suffix(".historico.json").write_text(json.dumps(historico,indent=2),encoding="utf-8")
    print("Checkpoint:",a.saida,"| conjunto test nao utilizado; avaliar separadamente.")

if __name__=="__main__":main()
