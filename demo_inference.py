#!/usr/bin/env python3
"""
Script de demostración para realizar inferencias con el modelo entrenado

Uso:
    python demo_inference.py
    python demo_inference.py --image path/to/image.jpg
    python demo_inference.py --demo-dir path/to/demo/images/
"""

import argparse
import sys
import os
from PIL import Image

# Agregar src al path
sys.path.append('src')

from inference import AlzheimerPredictor, create_demo_predictions
from config import MODEL_SAVE_PATH


def main():
    parser = argparse.ArgumentParser(description='Demo de inferencia para detección de Alzheimer')
    parser.add_argument('--image', type=str, help='Ruta a una imagen específica para predecir')
    parser.add_argument('--demo-dir', type=str, help='Directorio con imágenes de demostración')
    parser.add_argument('--model', type=str, default=MODEL_SAVE_PATH, 
                       help='Ruta al modelo entrenado')
    
    args = parser.parse_args()
    
    # Verificar que el modelo existe
    if not os.path.exists(args.model):
        print(f"❌ Modelo no encontrado: {args.model}")
        print("Primero ejecuta: python train_alzheimer_model.py")
        return
    
    print("🧠 DEMO - DETECCIÓN DE ALZHEIMER")
    print("=" * 40)
    
    # Inicializar predictor
    try:
        predictor = AlzheimerPredictor(model_path=args.model)
        print("✅ Modelo cargado exitosamente")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return
    
    # Caso 1: Imagen específica
    if args.image:
        if not os.path.exists(args.image):
            print(f"❌ Imagen no encontrada: {args.image}")
            return
        
        print(f"\n🔍 Analizando imagen: {args.image}")
        try:
            result = predictor.visualize_prediction(args.image)
            print("\n📊 Resultado:")
            print(f"   Predicción: {result['predicted_class']}")
            print(f"   Confianza: {result['confidence']:.1%}")
            print("\n📈 Probabilidades por clase:")
            for class_name, prob in result['probabilities'].items():
                print(f"   {class_name}: {prob:.1%}")
        except Exception as e:
            print(f"❌ Error procesando imagen: {e}")
        return
    
    # Caso 2: Directorio de demostración
    if args.demo_dir:
        if not os.path.exists(args.demo_dir):
            print(f"❌ Directorio no encontrado: {args.demo_dir}")
            return
        
        print(f"\n📁 Procesando directorio: {args.demo_dir}")
        try:
            results = create_demo_predictions(predictor, args.demo_dir)
            print("✅ Predicciones completadas")
        except Exception as e:
            print(f"❌ Error procesando directorio: {e}")
        return
    
    # Caso 3: Demo interactivo con dataset
    print("\n🎮 MODO DEMO INTERACTIVO")
    print("Cargando ejemplos del dataset...")
    
    try:
        from datasets import load_dataset
        ds = load_dataset("Falah/Alzheimer_MRI")
        
        # Tomar algunos ejemplos del test
        test_examples = [0, 10, 20, 30, 40]  # Índices de ejemplo
        
        for i, idx in enumerate(test_examples):
            if idx >= len(ds['test']):
                continue
                
            print(f"\n--- Ejemplo {i+1}/5 ---")
            sample = ds['test'][idx]
            true_label = ds['test'].features['label'].names[sample['label']]
            
            print(f"Etiqueta real: {true_label}")
            
            # Realizar predicción
            result = predictor.predict_single_image(sample['image'])
            
            print(f"Predicción: {result['predicted_class']}")
            print(f"Confianza: {result['confidence']:.1%}")
            
            # Verificar si es correcto
            correct = "✅" if result['predicted_class'] == true_label else "❌"
            print(f"Resultado: {correct}")
            
            # Mostrar probabilidades
            print("Probabilidades:")
            for class_name, prob in result['probabilities'].items():
                marker = "👉" if class_name == result['predicted_class'] else "  "
                print(f"  {marker} {class_name}: {prob:.1%}")
            
            # Preguntar si continuar
            if i < len(test_examples) - 1:
                response = input("\nPresiona Enter para continuar (o 'q' para salir): ")
                if response.lower() == 'q':
                    break
        
        print("\n🎉 Demo completado!")
        
    except Exception as e:
        print(f"❌ Error en demo interactivo: {e}")
        print("Asegúrate de tener conexión a internet para descargar el dataset")


def show_usage_examples():
    """Muestra ejemplos de uso"""
    print("\n📖 EJEMPLOS DE USO:")
    print("1. Demo interactivo:")
    print("   python demo_inference.py")
    print("\n2. Analizar imagen específica:")
    print("   python demo_inference.py --image mi_imagen.jpg")
    print("\n3. Procesar directorio completo:")
    print("   python demo_inference.py --demo-dir mi_directorio/")
    print("\n4. Usar modelo personalizado:")
    print("   python demo_inference.py --model mi_modelo.pt --image imagen.jpg")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Si no hay argumentos, mostrar ayuda
        print("🧠 SISTEMA DE DETECCIÓN DE ALZHEIMER")
        print("=" * 40)
        show_usage_examples()
        print("\n¿Quieres ejecutar el demo interactivo? (y/n): ", end="")
        response = input().lower()
        if response == 'y' or response == 'yes' or response == '':
            main()
    else:
        main()