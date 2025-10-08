#!/usr/bin/env python3
"""
Script para generar presentación completa del proyecto de detección de Alzheimer
Incluye análisis del dataset, visualizaciones de data augmentation y resultados
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import warnings
from tqdm import tqdm

# Agregar src al path
sys.path.append('src')

try:
    from config import *
    from data_utils import load_and_prepare_data, get_transforms
    from model import create_model
    from evaluation import create_presentation_plots
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate de que el directorio 'src' existe y contiene los módulos necesarios")
    sys.exit(1)

# Configurar warnings y matplotlib
warnings.filterwarnings('ignore', category=UserWarning)
plt.style.use('default')


def print_header():
    """Imprime el encabezado de la presentación"""
    print("🎬 GENERANDO PRESENTACIÓN COMPLETA")
    print("=" * 50)
    print("🎯 RESUMEN EJECUTIVO - DETECCIÓN DE ALZHEIMER")
    print("=" * 60)
    print("Objetivo: Detectar niveles de demencia en imágenes de resonancia magnética")
    print("Dataset: Falah/Alzheimer_MRI (Hugging Face)")
    print("Modelo: ResNet18 preentrenado en ImageNet")
    print("Clases: NonDemented, VeryMildDemented, MildDemented, ModerateDemented")
    print("Técnica: Transfer Learning + Data Augmentation")
    print("Framework: PyTorch")
    print("=" * 60)


def analyze_dataset(ds, label_names):
    """Analiza y muestra estadísticas del dataset"""
    print("\n📊 ANÁLISIS DEL DATASET")
    print("-" * 30)
    print("Cargando dataset...")
    
    # Información básica
    train_size = len(ds['train'])
    test_size = len(ds['test'])
    total_size = train_size + test_size
    
    print(f"Clases encontradas: {label_names}")
    print(f"Tamaño total del dataset: {total_size:,} imágenes")
    print(f"  - Entrenamiento: {train_size:,} imágenes")
    print(f"  - Test: {test_size:,} imágenes")
    print(f"Número de clases: {len(label_names)}")
    print(f"Clases: {', '.join(label_names)}")
    
    # Distribución en entrenamiento
    train_labels = [ds['train'][i]['label'] for i in range(len(ds['train']))]
    train_df = pd.DataFrame({'label': train_labels})
    train_counts = train_df['label'].value_counts().sort_index()
    
    print(f"\nDistribución en entrenamiento:")
    for idx, count in train_counts.items():
        if idx < len(label_names):
            percentage = (count / train_size) * 100
            print(f"  {label_names[idx]}: {count} ({percentage:.1f}%)")


def demonstrate_data_augmentation(ds, label_names):
    """Demuestra las técnicas de data augmentation aplicadas"""
    print("\n🔄 TÉCNICAS DE DATA AUGMENTATION")
    print("-" * 35)
    print("Técnicas aplicadas:")
    print("✓ Conversión a 3 canales (RGB)")
    print("✓ Redimensionamiento a 224x224")
    print("✓ Rotación aleatoria (±10°)")
    print("✓ Traslación aleatoria (±5%)")
    print("✓ Ajuste de brillo/contraste (±10%)")
    print("✓ Ruido gaussiano suave (σ=0.02)")
    print("✓ Normalización ImageNet")
    
    # Crear transformaciones para demostración
    orig_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224), InterpolationMode.BILINEAR),
        transforms.ToTensor()
    ])
    
    # Obtener transformaciones del proyecto
    train_pipeline, eval_pipeline = get_transforms()
    
    def denorm_imagenet(tensor):
        """Desnormaliza tensor usando estadísticas de ImageNet"""
        mean = torch.tensor(IMAGENET_MEAN, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        return torch.clamp(tensor * std + mean, 0, 1)
    
    # Mostrar ejemplos de augmentación
    try:
        split_name = list(ds.keys())[0]
        sample = ds[split_name][0]
        pil_image = sample['image']
        label_id = sample['label']
        label_name = label_names[label_id] if label_id < len(label_names) else f"Clase_{label_id}"
        
        print(f"\nEjemplo de augmentación - Clase: {label_name}")
        
        # Crear figura
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Imagen original
        orig_tensor = orig_transform(pil_image)
        axes[0].imshow(np.transpose(orig_tensor.numpy(), (1, 2, 0)))
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis('off')
        
        # Imagen sin augmentación (eval)
        eval_tensor = eval_pipeline(pil_image)
        eval_vis = denorm_imagenet(eval_tensor)
        axes[1].imshow(np.transpose(eval_vis.numpy(), (1, 2, 0)))
        axes[1].set_title("Sin Augmentación", fontsize=12)
        axes[1].axis('off')
        
        # Imágenes con augmentación
        for i in range(6):
            aug_tensor = train_pipeline(pil_image)
            aug_vis = denorm_imagenet(aug_tensor)
            axes[i + 2].imshow(np.transpose(aug_vis.numpy(), (1, 2, 0)))
            axes[i + 2].set_title(f"Augmentación {i + 1}", fontsize=12)
            axes[i + 2].axis('off')
        
        plt.suptitle(f"Técnicas de Data Augmentation - {label_name}", fontsize=14, y=1.02)
        plt.tight_layout()
        
        # Guardar figura
        os.makedirs("results", exist_ok=True)
        plt.savefig("results/data_augmentation_demo.png", dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Visualización de augmentación completada")
        
    except Exception as e:
        print(f"⚠️  Error en visualización de augmentación: {e}")


def show_model_architecture():
    """Muestra información sobre la arquitectura del modelo"""
    print("\n🏗️ ARQUITECTURA DEL MODELO")
    print("-" * 30)
    print("Modelo base: ResNet18 (preentrenado en ImageNet)")
    print("Modificaciones:")
    print("  ✓ Capa final adaptada para 4 clases")
    print("  ✓ Transfer Learning con fine-tuning")
    print("  ✓ Optimizador: AdamW")
    print("  ✓ Scheduler: ReduceLROnPlateau")
    print("  ✓ Pérdida: CrossEntropyLoss con pesos por clase")
    
    try:
        # Crear modelo para mostrar información
        model = create_model(num_classes=4)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nParámetros del modelo:")
        print(f"  Total: {total_params:,}")
        print(f"  Entrenables: {trainable_params:,}")
        print(f"  Dispositivo: {DEVICE}")
        
    except Exception as e:
        print(f"⚠️  Error obteniendo información del modelo: {e}")


def show_training_configuration():
    """Muestra la configuración de entrenamiento"""
    print("\n⚙️ CONFIGURACIÓN DE ENTRENAMIENTO")
    print("-" * 35)
    print(f"Épocas: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Weight Decay: {WEIGHT_DECAY}")
    print(f"Semilla aleatoria: {SEED}")
    print(f"Validación: {VAL_SPLIT_SIZE * 100:.0f}% del conjunto de entrenamiento")
    
    # Información del sistema
    print(f"\nSistema:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Dispositivo: {DEVICE}")


def show_expected_results():
    """Muestra los resultados esperados del entrenamiento"""
    print("\n📈 RESULTADOS ESPERADOS")
    print("-" * 25)
    print("Métricas objetivo:")
    print("  🎯 Accuracy: > 85%")
    print("  🎯 F1-Score (macro): > 0.80")
    print("  🎯 AUC-ROC (macro): > 0.90")
    
    print("\nTécnicas para mejorar rendimiento:")
    print("  ✓ Data Augmentation")
    print("  ✓ Class Weighting (manejo de desbalance)")
    print("  ✓ Transfer Learning")
    print("  ✓ Mixed Precision Training")
    print("  ✓ Learning Rate Scheduling")


def check_requirements():
    """Verifica que todos los requisitos estén instalados"""
    print("\n🔍 VERIFICACIÓN DE REQUISITOS")
    print("-" * 30)
    
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pandas', 'matplotlib',
        'sklearn', 'datasets', 'tqdm', 'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package} (faltante)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Paquetes faltantes: {', '.join(missing_packages)}")
        print("Instalar con: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ Todos los requisitos están instalados")
        return True


def main():
    """Función principal"""
    # Imprimir encabezado
    print_header()
    
    # Verificar requisitos
    if not check_requirements():
        print("\n❌ Faltan dependencias. Por favor instálalas antes de continuar.")
        return
    
    try:
        # Cargar y analizar dataset
        ds, label_names = load_and_prepare_data()
        analyze_dataset(ds, label_names)
        
        # Demostrar data augmentation
        demonstrate_data_augmentation(ds, label_names)
        
        # Mostrar arquitectura del modelo
        show_model_architecture()
        
        # Mostrar configuración de entrenamiento
        show_training_configuration()
        
        # Mostrar resultados esperados
        show_expected_results()
        
        print("\n" + "=" * 60)
        print("🎉 PRESENTACIÓN COMPLETADA")
        print("=" * 60)
        print("\nPróximos pasos:")
        print("1. Ejecutar entrenamiento: python train_alzheimer_model.py")
        print("2. Ver demo de inferencia: python demo_inference.py")
        print("3. Revisar resultados en: results/")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error durante la presentación: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()