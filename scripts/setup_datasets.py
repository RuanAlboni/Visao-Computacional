"""
Script para criar estrutura de diretórios dos datasets
"""
from pathlib import Path

def create_dataset_structure():
    """Cria estrutura de diretórios para cada produto"""
    
    # Diretório base
    DATA_DIR = Path("data/raw")
    
    # Produtos
    products = ["tomato", "apple", "strawberry"]
    
    # Splits
    splits = ["train", "val", "test"]
    
    for product in products:
        for split in splits:
            # Criar diretórios de imagens e labels
            (DATA_DIR / product / "images" / split).mkdir(parents=True, exist_ok=True)
            (DATA_DIR / product / "labels" / split).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Estrutura criada para: {product}")
    
    print("\n✅ Estrutura completa criada!")
    print(f"📁 Diretório base: {DATA_DIR.absolute()}")

if __name__ == "__main__":
    create_dataset_structure()
