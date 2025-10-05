#!/usr/bin/env python3
"""
Script principal para entrenar el modelo de detección de Alzheimer

Uso:
    python train_alzheimer_model.py
    
Este script:
1. Carga y prepara el dataset de Alzheimer MRI
2. Crea los splits de entrenamiento/validación/test
3. Entrena un modelo ResNet18 preentrenado
4. Evalúa el modelo y genera visualizaciones
5. Guarda el modelo entrenado y los resultados
"""

import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import warnings

# Agregar src al path
sys.path.append('src')

from config import *
from data_utils import (
    load_and_prepare_data, create_metadata_dataframe, analyze_data_quality,
    create_data_splits, create_data_loaders, get_transforms, visualize_augmentations
)
from model import create_model, calculate_class_weights, create_optimizer_scheduler, train_model
from evaluation import create_presentation_plots, analyze_misclassifications
from inference import AlzheimerPredictor

# Configurar warnings
warnings.filterwarnings('ignore', category=UserWarning)
plt.style.use('default')


def set_random_seeds(seed=SEED):
    """Configura las semillas aleatorias para reproducibilidad"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_system_info():
    """Muestra información del sistema"""
    print("=" * 60)
    print("SISTEMA DE DETECCIÓN DE ALZHEIMER CON RESNET18")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Dispositivo seleccionado: {DEVICE}")
    print(f"Semilla aleatoria: {SEED}")
    print("=" * 60)


def main():
    """Función principal"""
    # Configuración inicial
    print_system_info()
    set_random_seeds(SEED)
    
    # Crear directorios necesarios
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    print("\n🔄 PASO 1: Cargando y preparando datos...")
    # Cargar dataset
    ds, label_names = load_and_prepare_data()
    print(f"✅ Dataset cargado. Clases: {label_names}")
    
    # Crear DataFrame de metadatos
    split_name = list(ds.keys())[0]
    df_metadata = create_metadata_dataframe(ds, split_name, label_names)
    print(f"✅ Metadatos creados: {len(df_metadata)} imágenes")
    
    # Mostrar distribución inicial
    print("\nDistribución de clases en el dataset completo:")
    class_counts = df_metadata["label"].value_counts().sort_index()
    pct = (class_counts / len(df_metadata) * 100).round(2)
    for label, count, percentage in zip(class_counts.index, class_counts.values, pct.values):
        print(f"  {label}: {count} ({percentage}%)")
    
    print("\n🔄 PASO 2: Análisis de calidad de datos...")
    # Análisis de calidad (opcional, puede tomar tiempo)
    try:
        stats_df, corrupt_idx, low_var_idx = analyze_data_quality(ds, split_name)
        print(f"✅ Análisis completado. Imágenes problemáticas: {len(corrupt_idx + low_var_idx)}")
    except Exception as e:
        print(f"⚠️  Análisis de calidad omitido: {e}")
    
    print("\n🔄 PASO 3: Creando splits de datos...")
    # Crear splits
    hf_splits, indices_splits, class_distributions = create_data_splits(ds, label_names)
    print("✅ Splits creados exitosamente")
    
    print("\n🔄 PASO 4: Preparando transformaciones y DataLoaders...")
    # Obtener transformaciones
    transforms_tuple = get_transforms()
    print("✅ Transformaciones configuradas")
    
    # Crear DataLoaders
    train_loader, val_loader, test_loader = create_data_loaders(
        hf_splits, indices_splits, transforms_tuple
    )
    print("✅ DataLoaders creados")
    
    print("\n🔄 PASO 5: Visualizando augmentaciones (opcional)...")
    # Mostrar ejemplos de augmentaciones
    try:
        print("Mostrando ejemplos de augmentaciones...")
        visualize_augmentations(ds, label_names, num_examples=2)
        print("✅ Visualizaciones mostradas")
    except Exception as e:
        print(f"⚠️  Visualizaciones omitidas: {e}")
    
    print("\n🔄 PASO 6: Creando modelo...")
    # Crear modelo
    model = create_model(num_classes=len(label_names))
    print("✅ Modelo ResNet18 creado")
    
    print("\n🔄 PASO 7: Configurando entrenamiento...")
    # Calcular pesos por clase
    hf_train, _ = hf_splits
    train_idx, _, _ = indices_splits
    class_weights = calculate_class_weights(hf_train, train_idx, label_names)
    
    # Crear optimizador y scheduler
    criterion, optimizer, scheduler = create_optimizer_scheduler(model, class_weights)
    print("✅ Optimizador y scheduler configurados")
    
    print(f"\n🚀 PASO 8: Iniciando entrenamiento ({EPOCHS} épocas)...")
    print("Este proceso puede tomar varios minutos...")
    
    # Entrenar modelo
    training_history, best_f1, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        label_names=label_names,
        epochs=EPOCHS,
        save_path=MODEL_SAVE_PATH
    )
    
    print(f"✅ Entrenamiento completado!")
    print(f"   Mejor F1-Macro: {best_f1:.3f} (época {best_epoch})")
    print(f"   Modelo guardado en: {MODEL_SAVE_PATH}")
    
    print("\n🔄 PASO 9: Evaluando en conjunto de test...")
    # Cargar mejor modelo para evaluación final
    from model import evaluate_model
    
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    test_metrics = evaluate_model(model, test_loader, criterion, DEVICE, label_names)
    
    print("✅ Evaluación en test completada:")
    print(f"   Test Accuracy: {test_metrics['accuracy']:.3f}")
    print(f"   Test F1-Macro: {test_metrics['f1_macro']:.3f}")
    print(f"   Test AUC-Macro: {test_metrics['auc_macro']:.3f}")
    
    print("\n🔄 PASO 10: Generando visualizaciones para presentación...")
    # Crear todas las visualizaciones
    create_presentation_plots(
        test_metrics=test_metrics,
        training_history=training_history,
        label_names=label_names,
        class_distributions=class_distributions,
        results_dir=RESULTS_PATH
    )
    print("✅ Visualizaciones guardadas en results/")
    
    print("\n🔄 PASO 11: Análisis de errores...")
    # Analizar clasificaciones erróneas
    try:
        misclassified_indices, confusion_pairs = analyze_misclassifications(
            test_metrics, label_names, top_n=5
        )
        print("✅ Análisis de errores completado")
    except Exception as e:
        print(f"⚠️  Análisis de errores omitido: {e}")
    
    print("\n🔄 PASO 12: Probando sistema de inferencia...")
    # Probar el predictor
    try:
        predictor = AlzheimerPredictor(model_path=MODEL_SAVE_PATH, label_names=label_names)
        print("✅ Sistema de inferencia inicializado correctamente")
        
        # Probar con una imagen del dataset de test
        test_sample = ds['test'][0]['image']
        result = predictor.predict_single_image(test_sample)
        print(f"   Predicción de prueba: {result['predicted_class']} (confianza: {result['confidence']:.3f})")
        
    except Exception as e:
        print(f"⚠️  Error en sistema de inferencia: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
    print("=" * 60)
    print("\nArchivos generados:")
    print(f"📁 Modelo entrenado: {MODEL_SAVE_PATH}")
    print(f"📁 Resultados y gráficos: {RESULTS_PATH}")
    print(f"📁 Resumen detallado: {RESULTS_PATH}results_summary.txt")
    
    print("\nPara usar el modelo entrenado:")
    print("1. Ejecutar: python demo_inference.py")
    print("2. O importar: from src.inference import AlzheimerPredictor")
    
    print("\nMétricas finales:")
    print(f"🎯 Test Accuracy: {test_metrics['accuracy']:.1%}")
    print(f"🎯 Test F1-Macro: {test_metrics['f1_macro']:.3f}")
    print(f"🎯 Test AUC-Macro: {test_metrics['auc_macro']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Entrenamiento interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)