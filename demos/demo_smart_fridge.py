"""
Demo: Smart Fridge Vision
Sistema de detecção de frutas para geladeiras inteligentes
"""

from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def demo_inference(image_path, model_path):
    """
    Demo de inferência em uma imagem
    """
    # Carregar modelo
    print(f"🔍 Carregando modelo: {model_path}")
    model = YOLO(model_path)
    
    # Carregar imagem
    print(f"📷 Processando imagem: {image_path}")
    image = cv2.imread(image_path)
    
    # Fazer inferência
    results = model.predict(
        source=image_path,
        save=False,
        show=False,
        conf=0.5
    )
    
    # Mostrar resultados
    result = results[0]
    
    print(f"\n✅ Detecções encontradas: {len(result.boxes)}")
    
    if len(result.boxes) > 0:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls]
            
            print(f"  - {class_name}: {conf:.2%} confiança")
    
    # Plotar resultados
    plt.figure(figsize=(12, 8))
    plt.imshow(result.plot())
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('demo_result.png', dpi=150)
    print(f"\n📊 Resultado salvo em: demo_result.png")
    
    return result

if __name__ == "__main__":
    # Configurações
    MODEL_PATH = "runs/detect/models/yolo/tomato_v1-5/weights/best.pt"
    IMAGE_PATH = "data/raw/tomato/images/train/000000000072.jpg"
    
    # Rodar demo
    demo_inference(IMAGE_PATH, MODEL_PATH)
