#!/usr/bin/env python3
"""
Script de demostración para presentaciones
Genera visualizaciones y análisis completos del modelo de Alzheimer
"""

import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from torchvision import transforms

# Configurar matplotlib para presentaciones
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11

# Agregar src al path
sys.path.append('src')

from config import *
from data_utils import load_and_prepare_data, create_data_splits, get_transforms
from inference import AlzheimerPredictor
from evaluation import *


def create_presentation_summary():
    """Crea un resumen ejecutivo para la presentación"""
    print("🎯 RESUMEN EJECUTIVO - DETECCIÓN DE ALZHEIMER")
    print("=" * 60)
    
    # Información del proyecto
    project_info = {
        "Objetivo": "Detectar niveles de demencia en imágenes de resonancia magnética",
        "Dataset": "Falah/Alzheimer_MRI (Hugging Face)",
        "Modelo": "ResNet18 preentrenado en ImageNet",
        "Clases": ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"],
        "Técnica": "Transfer Learning + Data Augmentation",
        "Framework": "PyTorch"
    }
    
    for key, value in project_info.items():
        if isinstance(value, list):
            print(f"{key}: {', '.join(value)}")
        else:
            print(f"{key}: {value}")
    
    print("=" * 60)


def analyze_dataset_characteristics():
    """Analiza las características del dataset"""
    print("\n📊 ANÁLISIS DEL DATASET")
    print("-" * 30)
    
    # Cargar datos
    ds, label_names = load_and_prepare_data()
    
    # Estadísticas básicas
    train_size = len(ds['train'])
    test_size = len(ds['test'])
    total_size = train_size + test_size
    
    print(f"Tamaño total del dataset: {total_size:,} imágenes")
    print(f"  - Entrenamiento: {train_size:,} imágenes")
    print(f"  - Test: {test_size:,} imágenes")
    print(f"Número de clases: {len(label_names)}")
    print(f"Clases: {', '.join(label_names)}")
    
    # Distribución por clase en entrenamiento
    train_labels = [ds['train'][i]['label'] for i in range(len(ds['train']))]
    train_dist = pd.Series(train_labels).value_counts().sort_index()
    train_dist.index = [label_names[i] for i in train_dist.index]
    
    print(f"\nDistribución en entrenamiento:")
    for label, count in train_dist.items():
        pct = (count / train_size) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")
    
    # Visualizar distribución
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = plt.bar(train_dist.index, train_dist.values, color=colors, alpha=0.8)
    
    # Agregar valores en las barras
    for bar, value in zip(bars, train_dist.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{value:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Distribución de Clases en el Dataset de Entrenamiento', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Clase de Demencia', fontsize=14)
    plt.ylabel('Número de Imágenes', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Guardar
    os.makedirs('results/presentation', exist_ok=True)
    plt.savefig('results/presentation/dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return ds, label_names


def demonstrate_data_augmentation(ds, label_names):
    """Demuestra las técnicas de data augmentation"""
    print("\n🔄 TÉCNICAS DE DATA AUGMENTATION")
    print("-" * 35)
    
    # Obtener transformaciones
    train_pipeline, eval_pipeline = get_transforms()
    
    print("Técnicas aplicadas:")
    print("✓ Conversión a 3 canales (RGB)")
    print("✓ Redimensionamiento a 224x224")
    print("✓ Rotación aleatoria (±10°)")
    print("✓ Traslación aleatoria (±5%)")
    print("✓ Ajuste de brillo/contraste (±10%)")
    print("✓ Ruido gaussiano suave (σ=0.02)")
    print("✓ Normalización ImageNet")
    
    # Mostrar ejemplo visual
    def denorm_imgnet(t):
        mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device).view(3,1,1)
        std = torch.tensor(IMAGENET_STD, dtype=t.dtype, device=t.device).view(3,1,1)
        return torch.clamp(t * std + mean, 0, 1)
    
    # Tomar una imagen de ejemplo
    sample = ds['train'][0]
    pil_img = sample['image']
    true_label = label_names[sample['label']]
    
    # Crear múltiples versiones
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Original
    orig_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    orig_tensor = orig_transform(pil_img)
    axes[0].imshow(np.transpose(orig_tensor.numpy(), (1, 2, 0)))
    axes[0].set_title('Original', fontweight='bold')
    axes[0].axis('off')
    
    # Versiones aumentadas
    for i in range(1, 8):
        aug_tensor = train_pipeline(pil_img)
        aug_vis = denorm_imgnet(aug_tensor)
        axes[i].imshow(np.transpose(aug_vis.numpy(), (1, 2, 0)))
        axes[i].set_title(f'Augmentación {i}', fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle(f'Data Augmentation - Clase: {true_label}', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('results/presentation/data_augmentation.png', dpi=300, bbox_inches='tight')
    plt.show()


def show_model_architecture():
    """Muestra información sobre la arquitectura del modelo"""
    print("\n🏗️  ARQUITECTURA DEL MODELO")
    print("-" * 28)
    
    print("Modelo base: ResNet18")
    print("✓ Preentrenado en ImageNet (1M+ imágenes)")
    print("✓ 18 capas profundas")
    print("✓ Conexiones residuales")
    print("✓ Capa final modificada para 4 clases")
    print(f"✓ ~11M parámetros entrenables")
    
    # Crear diagrama conceptual
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Definir bloques
    blocks = [
        ("Input\n224×224×3", 0, 0.8, 0.15, 0.1, '#FFE5E5'),
        ("Conv1\n112×112×64", 0.2, 0.8, 0.15, 0.1, '#E5F3FF'),
        ("ResBlock1\n56×56×64", 0.4, 0.8, 0.15, 0.1, '#E5FFE5'),
        ("ResBlock2\n28×28×128", 0.6, 0.8, 0.15, 0.1, '#FFFEE5'),
        ("ResBlock3\n14×14×256", 0.4, 0.5, 0.15, 0.1, '#F0E5FF'),
        ("ResBlock4\n7×7×512", 0.6, 0.5, 0.15, 0.1, '#FFE5F0'),
        ("GlobalPool\n1×1×512", 0.2, 0.2, 0.15, 0.1, '#E5FFFF'),
        ("FC Layer\n4 classes", 0.4, 0.2, 0.15, 0.1, '#FFF5E5'),
        ("Output\nProbabilities", 0.6, 0.2, 0.15, 0.1, '#F5FFE5')
    ]
    
    # Dibujar bloques
    for name, x, y, w, h, color in blocks:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, name, ha='center', va='center', 
                fontweight='bold', fontsize=10)
    
    # Dibujar flechas
    arrow_props = dict(arrowstyle='->', lw=2, color='darkblue')
    arrows = [
        ((0.15, 0.85), (0.2, 0.85)),   # Input -> Conv1
        ((0.35, 0.85), (0.4, 0.85)),   # Conv1 -> ResBlock1
        ((0.55, 0.85), (0.6, 0.85)),   # ResBlock1 -> ResBlock2
        ((0.67, 0.8), (0.47, 0.6)),    # ResBlock2 -> ResBlock3
        ((0.55, 0.55), (0.6, 0.55)),   # ResBlock3 -> ResBlock4
        ((0.67, 0.5), (0.27, 0.3)),    # ResBlock4 -> GlobalPool
        ((0.35, 0.25), (0.4, 0.25)),   # GlobalPool -> FC
        ((0.55, 0.25), (0.6, 0.25)),   # FC -> Output
    ]
    
    for start, end in arrows:
        ax.annotate('', xy=end, xytext=start, arrowprops=arrow_props)
    
    ax.set_xlim(-0.05, 0.8)
    ax.set_ylim(0.1, 0.95)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Arquitectura ResNet18 para Detección de Alzheimer', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('results/presentation/model_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()


def load_and_show_results():
    """Carga y muestra los resultados del entrenamiento"""
    print("\n📈 RESULTADOS DEL MODELO")
    print("-" * 25)
    
    # Verificar si existen resultados
    results_summary_path = os.path.join(RESULTS_PATH, 'results_summary.txt')
    
    if not os.path.exists(results_summary_path):
        print("⚠️  Resultados no encontrados. Ejecuta primero: python train_alzheimer_model.py")
        return
    
    # Leer resumen
    with open(results_summary_path, 'r', encoding='utf-8') as f:
        content = f.read()
        print(content)
    
    # Mostrar gráficos si existen
    plots = [
        ('confusion_matrix_normalized.png', 'Matriz de Confusión'),
        ('roc_curves.png', 'Curvas ROC'),
        ('training_history.png', 'Historial de Entrenamiento')
    ]
    
    for plot_file, title in plots:
        plot_path = os.path.join(RESULTS_PATH, plot_file)
        if os.path.exists(plot_path):
            print(f"\n📊 {title}:")
            # Aquí podrías mostrar la imagen si estás en un notebook
            print(f"   Guardado en: {plot_path}")


def demonstrate_inference():
    """Demuestra el sistema de inferencia"""
    print("\n🔮 DEMOSTRACIÓN DE INFERENCIA")
    print("-" * 32)
    
    # Verificar modelo
    if not os.path.exists(MODEL_SAVE_PATH):
        print("⚠️  Modelo no encontrado. Ejecuta primero: python train_alzheimer_model.py")
        return
    
    try:
        # Inicializar predictor
        predictor = AlzheimerPredictor(MODEL_SAVE_PATH)
        print("✅ Modelo cargado exitosamente")
        
        # Cargar dataset para ejemplos
        from datasets import load_dataset
        ds = load_dataset("Falah/Alzheimer_MRI")
        
        # Realizar predicciones en algunos ejemplos
        print("\n🧪 Ejemplos de predicción:")
        
        test_indices = [0, 50, 100, 150, 200]  # Diferentes ejemplos
        results = []
        
        for i, idx in enumerate(test_indices):
            if idx >= len(ds['test']):
                continue
                
            sample = ds['test'][idx]
            true_label = ds['test'].features['label'].names[sample['label']]
            
            # Predicción
            result = predictor.predict_single_image(sample['image'])
            
            correct = result['predicted_class'] == true_label
            status = "✅" if correct else "❌"
            
            print(f"\nEjemplo {i+1}:")
            print(f"  Verdad: {true_label}")
            print(f"  Predicción: {result['predicted_class']}")
            print(f"  Confianza: {result['confidence']:.1%}")
            print(f"  Estado: {status}")
            
            results.append({
                'ejemplo': i+1,
                'verdad': true_label,
                'prediccion': result['predicted_class'],
                'confianza': result['confidence'],
                'correcto': correct
            })
        
        # Resumen de rendimiento
        accuracy = sum(r['correcto'] for r in results) / len(results)
        avg_confidence = sum(r['confianza'] for r in results) / len(results)
        
        print(f"\n📊 Resumen de ejemplos:")
        print(f"  Precisión: {accuracy:.1%}")
        print(f"  Confianza promedio: {avg_confidence:.1%}")
        
    except Exception as e:
        print(f"❌ Error en demostración: {e}")


def create_technical_summary():
    """Crea un resumen técnico detallado"""
    print("\n🔬 RESUMEN TÉCNICO")
    print("-" * 20)
    
    technical_details = {
        "Preprocesamiento": [
            "Conversión a escala de grises → RGB (3 canales)",
            "Redimensionamiento a 224×224 píxeles",
            "Normalización con estadísticas de ImageNet",
            "Data augmentation: rotación, traslación, ruido"
        ],
        "Arquitectura": [
            "ResNet18 preentrenado en ImageNet",
            "Transfer learning con fine-tuning",
            "Capa final adaptada para 4 clases",
            "~11M parámetros entrenables"
        ],
        "Entrenamiento": [
            f"Optimizador: AdamW (lr={LEARNING_RATE})",
            f"Función de pérdida: CrossEntropy con pesos por clase",
            f"Scheduler: ReduceLROnPlateau",
            f"Épocas: {EPOCHS}",
            "Early stopping basado en F1-macro"
        ],
        "Evaluación": [
            "Métricas: Accuracy, F1-macro, AUC-macro",
            "Validación estratificada (15% del train)",
            "Test set independiente",
            "Matriz de confusión y curvas ROC"
        ]
    }
    
    for category, details in technical_details.items():
        print(f"\n{category}:")
        for detail in details:
            print(f"  • {detail}")


def main():
    """Función principal de la presentación"""
    print("🎬 GENERANDO PRESENTACIÓN COMPLETA")
    print("=" * 50)
    
    # 1. Resumen ejecutivo
    create_presentation_summary()
    
    # 2. Análisis del dataset
    ds, label_names = analyze_dataset_characteristics()
    
    # 3. Data augmentation
    demonstrate_data_augmentation(ds, label_names)
    
    # 4. Arquitectura del modelo
    show_model_architecture()
    
    # 5. Resultados
    load_and_show_results()
    
    # 6. Demostración de inferencia
    demonstrate_inference()
    
    # 7. Resumen técnico
    create_technical_summary()
    
    print("\n" + "=" * 50)
    print("🎉 PRESENTACIÓN COMPLETADA")
    print("=" * 50)
    print("\nArchivos generados en results/presentation/:")
    print("📊 dataset_distribution.png")
    print("🔄 data_augmentation.png") 
    print("🏗️  model_architecture.png")
    print("\nPara ver todos los resultados:")
    print("📁 Revisar carpeta results/")
    print("=" * 50)


if __name__ == "__main__":
    main()