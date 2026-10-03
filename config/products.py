"""
Configurações específicas por produto
Fácil adicionar novos produtos aqui
"""
from dataclasses import dataclass
from typing import List

@dataclass
class ProductConfig:
    name: str
    classes: List[str]
    dataset_path: str
    model_name: str
    epochs: int = 100
    img_size: int = 640
    batch_size: int = 16

# Configuração de cada produto
PRODUCTS = {
    "tomato": ProductConfig(
        name="tomato",
        classes=["tomato_appropriate", "tomato_inappropriate"],
        dataset_path="data/raw/tomato",
        model_name="tomato_yolov8n.pt",
        epochs=100,
        img_size=640,
        batch_size=16
    ),
    
    "apple": ProductConfig(
        name="apple",
        classes=["apple_appropriate", "apple_inappropriate"],
        dataset_path="data/raw/apple",
        model_name="apple_yolov8n.pt",
        epochs=100,
        img_size=640,
        batch_size=16
    ),
    
    "strawberry": ProductConfig(
        name="strawberry",
        classes=["strawberry_appropriate", "strawberry_inappropriate"],
        dataset_path="data/raw/strawberry",
        model_name="strawberry_yolov8n.pt",
        epochs=100,
        img_size=640,
        batch_size=16
    )
}

def get_product_config(product_name: str) -> ProductConfig:
    """Retorna config de um produto específico"""
    if product_name not in PRODUCTS:
        raise ValueError(f"Produto {product_name} não configurado")
    return PRODUCTS[product_name]