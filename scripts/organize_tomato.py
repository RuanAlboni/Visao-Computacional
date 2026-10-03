"""
Script para organizar dataset LaboroTomato no formato YOLO
"""
from pathlib import Path
import shutil

def organize_tomato_dataset():
    source = Path("data/raw/tomato_temp")
    target = Path("data/raw/tomato")
    
    # Criar estrutura target
    for split in ["train", "val", "test"]:
        (target / "images" / split).mkdir(parents=True, exist_ok=True)
        (target / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Verificar estrutura do source
    print(f"🔍 Estrutura do dataset: {source}")
    print(f"📁 Conteúdo: {list(source.iterdir())}")
    
    # Caso 1: Se tiver images/ e labels/ na raiz
    if (source / "images").exists() and (source / "labels").exists():
        print("✅ Estrutura detectada: images/ + labels/")
        
        # Mover tudo para train (depois dividimos)
        images = list((source / "images").glob("*.jpg")) + list((source / "images").glob("*.png"))
        labels = list((source / "labels").glob("*.txt"))
        
        print(f"📊 Imagens encontradas: {len(images)}")
        print(f"📊 Labels encontrados: {len(labels)}")
        
        for img_path in images:
            shutil.copy(img_path, target / "images" / "train" / img_path.name)
        
        for lbl_path in labels:
            shutil.copy(lbl_path, target / "labels" / "train" / lbl_path.name)
    
    # Caso 2: Se tiver train/ e val/
    elif (source / "train").exists():
        print("✅ Estrutura detectada: train/ + val/")
        
        # Copiar train
        if (source / "train" / "images").exists():
            for img_path in (source / "train" / "images").glob("*.jpg"):
                shutil.copy(img_path, target / "images" / "train" / img_path.name)
        
        if (source / "train" / "labels").exists():
            for lbl_path in (source / "train" / "labels").glob("*.txt"):
                shutil.copy(lbl_path, target / "labels" / "train" / lbl_path.name)
        
        # Copiar val
        if (source / "val" / "images").exists():
            for img_path in (source / "val" / "images").glob("*.jpg"):
                shutil.copy(img_path, target / "images" / "val" / img_path.name)
        
        if (source / "val" / "labels").exists():
            for lbl_path in (source / "val" / "labels").glob("*.txt"):
                shutil.copy(lbl_path, target / "labels" / "val" / lbl_path.name)
    
    else:
        print("❌ Estrutura não reconhecida!")
        return
    
    print(f"\n✅ Dataset organizado em: {target.absolute()}")
    
    # Contar arquivos
    train_imgs = len(list((target / "images" / "train").glob("*.jpg")))
    val_imgs = len(list((target / "images" / "val").glob("*.jpg")))
    
    print(f"📊 Train: {train_imgs} imagens")
    print(f"📊 Val: {val_imgs} imagens")

if __name__ == "__main__":
    organize_tomato_dataset()
