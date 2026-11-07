"""Schemas unificados para todos os módulos de fruta."""
from dataclasses import dataclass, field
from enum import Enum

class FruitCondition(Enum):
    FRESH = "fresh"
    ROTTEN = "rotten"
    UNKNOWN = "unknown"

@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

@dataclass
class FruitDetection:
    fruit_type: str
    condition: FruitCondition
    confidence: float
    detection_confidence: float
    bbox: BoundingBox
    method: str
    
    @property
    def is_appropriate(self) -> bool:
        return self.condition == FruitCondition.FRESH

@dataclass
class FruitResult:
    image_path: str
    detections: list = field(default_factory=list)
    
    @property
    def total_fruits(self) -> int:
        return len(self.detections)
    
    @property
    def appropriate_count(self) -> int:
        return sum(1 for d in self.detections if d.is_appropriate)
    
    @property
    def inappropriate_count(self) -> int:
        return sum(1 for d in self.detections if d.condition == FruitCondition.ROTTEN)
    
    def to_dict(self) -> dict:
        return {
            "image": self.image_path,
            "total": self.total_fruits,
            "appropriate": self.appropriate_count,
            "inappropriate": self.inappropriate_count,
            "detections": [
                {
                    "fruit": d.fruit_type,
                    "condition": d.condition.value,
                    "confidence": round(d.confidence, 3),
                    "bbox": [d.bbox.x, d.bbox.y, d.bbox.width, d.bbox.height],
                    "method": d.method,
                }
                for d in self.detections
            ],
        }
