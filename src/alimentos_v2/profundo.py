"""ResNet-18/CNN compartilhadas. Le o formato de checkpoint enviado pelo usuario."""
import numpy as np
from PIL import Image
from .catalogo import validar_classes

def criar(arquitetura,n,pretreinado=False):
    import torch
    from torch import nn
    from torchvision import models
    if arquitetura=="resnet18":
        m=models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretreinado else None)
        m.fc=nn.Linear(m.fc.in_features,n);return m
    if arquitetura=="cnn":
        class CNN(nn.Module):
            def __init__(self):
                super().__init__();blocos=[];canais=[3,32,64,128,192]
                for a,b in zip(canais[:-1],canais[1:]):
                    blocos += [nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(inplace=True),nn.MaxPool2d(2)]
                self.extrator=nn.Sequential(*blocos,nn.AdaptiveAvgPool2d(1))
                self.classificador=nn.Sequential(nn.Flatten(),nn.Dropout(0.3),nn.Linear(192,n))
            def forward(self,x):return self.classificador(self.extrator(x))
        return CNN()
    raise ValueError("Arquitetura deve ser resnet18 ou cnn")

def transformar(treino=False,tamanho=224,media=None,desvio=None):
    from torchvision import transforms as T
    ops=[T.RandomResizedCrop(tamanho,scale=(0.8,1)),T.RandomHorizontalFlip()] if treino else [T.Resize((tamanho,tamanho))]
    return T.Compose(ops+[T.ToTensor(),T.Normalize(media or [0.485,0.456,0.406],desvio or [0.229,0.224,0.225])])

class Classificador:
    def __init__(self,caminho,device="cpu"):
        import torch
        self.device=torch.device(device)
        ck=torch.load(caminho,map_location="cpu",weights_only=True)
        self.classes=validar_classes(ck["classes"])
        self.modelo=criar(ck["arquitetura"],len(self.classes))
        self.modelo.load_state_dict(ck["modelo"],strict=True)
        self.modelo.to(self.device).eval()
        self.transformar=transformar(tamanho=int(ck.get("tamanho",224)),media=ck.get("media"),desvio=ck.get("desvio"))

    def prever(self,rgb):
        import torch
        a=np.asarray(rgb)
        if a.size==0:raise ValueError("Recorte vazio")
        x=self.transformar(Image.fromarray(a.astype("uint8"))).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probs=self.modelo(x).float().softmax(dim=1)[0].cpu().tolist()
        return dict(zip(self.classes,probs))
