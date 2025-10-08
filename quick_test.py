#!/usr/bin/env python3
"""
Prueba rápida del sistema de detección de Alzheimer
Ejecuta una versión reducida del entrenamiento para verificar funcionamiento
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Agregar src al path
sys.path.append('src')

from config import *
from data_utils import load_and_prepare_data, create_data_splits, get_transforms, create_data_loaders
from model import create_model, calculate_class_weights, create_optimizer_scheduler
import torch

def quick_test():
    """Ejecuta una prueba rápida del sistema"""
    print("🧪 PRUEBA RÁPIDA DEL SISTEMA")
    print("=" * 40)
    
    # 1. Cargar datos
    print("📥 Cargando dataset...")
    ds, label_names = load_and_prepare_data()
    print(f"✅ Dataset cargado: {label_names}")
    
    # 2. Crear splits pequeños para prueba
    print("\n📊 Creando splits de datos...")
    hf_splits, indices_splits, class_distributions = create_data_splits(ds, label_names)
    
    # Reducir tamaños para prueba rápida
    train_idx, val_idx, test_idx = indices_splits
    train_idx_small = train_idx[:50]  # Solo 50 muestras para entrenar
    val_idx_small = val_idx[:20]      # Solo 20 para validar
    test_idx_small = test_idx[:10]    # Solo 10 para test
    
    print(f"✅ Splits reducidos: Train={len(train_idx_small)}, Val={len(val_idx_small)}, Test={len(test_idx_small)}")
    
    # 3. Crear DataLoaders
    print("\n🔄 Creando DataLoaders...")
    transforms_tuple = get_transforms()
    indices_splits_small = (train_idx_small, val_idx_small, test_idx_small)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        hf_splits, indices_splits_small, transforms_tuple
    )
    print("✅ DataLoaders creados")
    
    # 4. Crear modelo
    print("\n🏗️ Creando modelo...")
    model = create_model(num_classes=len(label_names))
    print("✅ Modelo ResNet18 creado")
    
    # 5. Configurar entrenamiento
    print("\n⚙️ Configurando entrenamiento...")
    hf_train, _ = hf_splits
    class_weights = calculate_class_weights(hf_train, train_idx_small, label_names)
    criterion, optimizer, scheduler = create_optimizer_scheduler(model, class_weights)
    print("✅ Optimizador configurado")
    
    # 6. Probar una época de entrenamiento
    print("\n🚀 Probando una época de entrenamiento...")
    from model import train_one_epoch, evaluate_model
    
    model.train()
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, use_amp=False)
    print(f"✅ Época completada - Loss: {train_loss:.4f}")
    
    # 7. Probar evaluación
    print("\n📊 Probando evaluación...")
    val_metrics = evaluate_model(model, val_loader, criterion, DEVICE, label_names)
    print(f"✅ Evaluación completada - Accuracy: {val_metrics['accuracy']:.3f}")
    
    # 8. Probar inferencia
    print("\n🔮 Probando sistema de inferencia...")
    from inference import AlzheimerPredictor
    
    # Guardar modelo temporalmente
    temp_model_path = "models/temp_test_model.pt"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), temp_model_path)
    
    # Probar predictor
    predictor = AlzheimerPredictor(temp_model_path, label_names)
    
    # Probar con una imagen del dataset
    test_sample = ds['test'][0]['image']
    result = predictor.predict_single_image(test_sample)
    
    print(f"✅ Predicción: {result['predicted_class']} (confianza: {result['confidence']:.3f})")
    
    # Limpiar archivo temporal
    if os.path.exists(temp_model_path):
        os.remove(temp_model_path)
    
    print("\n" + "=" * 40)
    print("🎉 ¡PRUEBA RÁPIDA EXITOSA!")
    print("=" * 40)
    print("El sistema está funcionando correctamente.")
    print("\nPara entrenar el modelo completo:")
    print("python3 train_alzheimer_model.py")
    print("=" * 40)

if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"\n❌ Error en la prueba: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)