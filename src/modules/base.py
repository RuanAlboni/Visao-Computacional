"""Contrato base para todos os módulos de fruta."""
from abc import ABC, abstractmethod
from src.schemas.fruit import FruitResult

class BaseFruitModule(ABC):
    fruit_type: str
    
    @abstractmethod
    def detect(self, image_path: str) -> FruitResult:
        ...
    
    @abstractmethod
    def detect_batch(self, image_paths: list) -> list:
        ...
    
    @abstractmethod
    def load_model(self) -> None:
        ...
