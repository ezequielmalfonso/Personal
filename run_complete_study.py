#!/usr/bin/env python3
"""
Estudio Completo de Detección de Alzheimer mediante Deep Learning
================================================================

Este script ejecuta un estudio completo de clasificación de tomografías cerebrales
para detectar diferentes niveles de Alzheimer usando múltiples arquitecturas de
deep learning.

Autor: Estudio de ML - Detección de Alzheimer
Fecha: 2024
"""

import os
import sys
import time
import json
import argparse
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

# Importar nuestros módulos
from alzheimer_detection_study import AlzheimerDatasetAnalyzer, DataPreprocessor, HFDataset
from models_and_training import AlzheimerClassifier, ModelEvaluator, compare_models, save_results_report

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def setup_environment():
    """Configura el entorno de trabajo"""
    print("🔧 Configurando entorno...")
    
    # Crear directorios necesarios
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Configurar device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Usando device: {device}")
    
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device

def calculate_class_weights(hf_train, train_idx, label_names):
    """Calcula pesos de clase para balancear el dataset"""
    print("⚖️  Calculando pesos de clase...")
    
    # Contar muestras por clase
    y_train = pd.Series([hf_train[i]["label"] for i in train_idx])
    class_counts = y_train.value_counts().sort_index()
    
    # Calcular pesos inversos
    total_samples = len(train_idx)
    num_classes = len(label_names)
    
    class_weights = []
    for i in range(num_classes):
        weight = total_samples / (num_classes * class_counts.get(i, 1))
        class_weights.append(weight)
    
    class_weights = np.array(class_weights)
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalizar
    
    print("📊 Distribución de clases:")
    for i, (name, count) in enumerate(zip(label_names, class_counts)):
        print(f"   {name}: {count} muestras (peso: {class_weights[i]:.3f})")
    
    return class_weights

def create_data_loaders(preprocessor, data_splits, batch_size=32, num_workers=2):
    """Crea los data loaders"""
    print(f"📦 Creando data loaders (batch_size={batch_size})...")
    
    train_idx, val_idx, test_idx, hf_train, hf_test = data_splits
    
    # Crear datasets
    train_ds = HFDataset(hf_train, train_idx, transform=preprocessor.train_pipeline)
    val_ds = HFDataset(hf_train, val_idx, transform=preprocessor.eval_pipeline)
    test_ds = HFDataset(hf_test, test_idx, transform=preprocessor.eval_pipeline)
    
    # Configurar data loaders
    pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    print(f"✅ Data loaders creados:")
    print(f"   Train: {len(train_ds)} muestras")
    print(f"   Val: {len(val_ds)} muestras")
    print(f"   Test: {len(test_ds)} muestras")
    
    return train_loader, val_loader, test_loader

def train_and_evaluate_model(model_name, label_names, train_loader, val_loader, test_loader, 
                           class_weights=None, epochs=20, pretrained=True):
    """Entrena y evalúa un modelo específico"""
    print(f"\n🚀 ENTRENANDO MODELO: {model_name.upper()}")
    print("="*60)
    
    start_time = time.time()
    
    # Crear modelo
    classifier = AlzheimerClassifier(
        num_classes=len(label_names),
        model_name=model_name,
        pretrained=pretrained
    )
    
    model = classifier.build_model()
    
    # Configurar entrenamiento
    classifier.setup_training(train_loader, val_loader, class_weights)
    
    # Crear evaluador
    evaluator = ModelEvaluator(label_names)
    
    # Entrenar
    best_f1, best_epoch = classifier.train(epochs, evaluator, save_best=True)
    
    # Cargar mejor modelo
    classifier.load_best_model()
    
    # Evaluar en test
    print(f"\n📊 Evaluando {model_name} en conjunto de test...")
    test_metrics, test_predictions = evaluator.evaluate_model(
        model, test_loader, classifier.criterion, classifier.device, "test"
    )
    
    training_time = time.time() - start_time
    
    # Mostrar resultados
    print(f"\n📈 RESULTADOS FINALES - {model_name.upper()}")
    print("-" * 50)
    print(f"🏆 Mejor F1-macro (val): {best_f1:.4f} en época {best_epoch}")
    print(f"⏱️  Tiempo de entrenamiento: {training_time/60:.1f} minutos")
    print(f"🎯 Accuracy (test): {test_metrics['accuracy']:.4f}")
    print(f"📊 F1-macro (test): {test_metrics['f1_macro']:.4f}")
    print(f"📊 F1-weighted (test): {test_metrics['f1_weighted']:.4f}")
    
    if not np.isnan(test_metrics.get('auc_macro', np.nan)):
        print(f"📈 AUC-macro (test): {test_metrics['auc_macro']:.4f}")
    
    print("\n📋 Reporte por clase:")
    print(test_metrics['classification_report'])
    
    # Generar visualizaciones
    y_true, y_pred, y_prob, _ = test_predictions
    
    # Matriz de confusión
    fig_cm = evaluator.plot_confusion_matrix(
        test_metrics['confusion_matrix'], 
        title=f"Matriz de Confusión - {model_name}",
        normalize=False
    )
    plt.savefig(f'figures/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig_cm_norm = evaluator.plot_confusion_matrix(
        test_metrics['confusion_matrix'], 
        title=f"Matriz de Confusión Normalizada - {model_name}",
        normalize=True
    )
    plt.savefig(f'figures/confusion_matrix_normalized_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Curvas ROC
    try:
        fig_roc = evaluator.plot_roc_curves(
            y_true, y_prob, 
            title=f"Curvas ROC - {model_name}"
        )
        plt.savefig(f'figures/roc_curves_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"⚠️  No se pudieron generar curvas ROC: {e}")
    
    # Curvas Precision-Recall
    try:
        fig_pr = evaluator.plot_precision_recall_curves(
            y_true, y_prob,
            title=f"Curvas Precisión-Recall - {model_name}"
        )
        plt.savefig(f'figures/precision_recall_{model_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"⚠️  No se pudieron generar curvas Precision-Recall: {e}")
    
    # Historial de entrenamiento
    classifier.plot_training_history()
    plt.savefig(f'figures/training_history_{model_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Mover modelo a carpeta de modelos
    os.rename(f'best_model_{model_name}.pt', f'models/best_model_{model_name}.pt')
    
    return {
        'test_metrics': test_metrics,
        'test_predictions': test_predictions,
        'training_time': training_time,
        'best_epoch': best_epoch,
        'best_f1_val': best_f1,
        'model_params': sum(p.numel() for p in model.parameters())
    }

def cross_validation_analysis(preprocessor, data_splits, label_names, k_folds=5):
    """Realiza análisis de validación cruzada"""
    print(f"\n🔄 ANÁLISIS DE VALIDACIÓN CRUZADA (K={k_folds})")
    print("="*60)
    
    train_idx, val_idx, test_idx, hf_train, hf_test = data_splits
    
    # Combinar train y val para CV
    all_train_idx = train_idx + val_idx
    y_all = np.array([hf_train[i]["label"] for i in all_train_idx])
    
    # Configurar validación cruzada estratificada
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    cv_results = {
        'resnet18': {'accuracy': [], 'f1_macro': [], 'f1_weighted': []},
        'custom_cnn': {'accuracy': [], 'f1_macro': [], 'f1_weighted': []}
    }
    
    for fold, (train_cv_idx, val_cv_idx) in enumerate(skf.split(all_train_idx, y_all)):
        print(f"\n📊 Fold {fold + 1}/{k_folds}")
        print("-" * 30)
        
        # Mapear índices
        train_fold_idx = [all_train_idx[i] for i in train_cv_idx]
        val_fold_idx = [all_train_idx[i] for i in val_cv_idx]
        
        # Crear data loaders para este fold
        train_fold_ds = HFDataset(hf_train, train_fold_idx, transform=preprocessor.train_pipeline)
        val_fold_ds = HFDataset(hf_train, val_fold_idx, transform=preprocessor.eval_pipeline)
        
        train_fold_loader = DataLoader(train_fold_ds, batch_size=32, shuffle=True, num_workers=2)
        val_fold_loader = DataLoader(val_fold_ds, batch_size=32, shuffle=False, num_workers=2)
        
        # Entrenar modelos rápidos para CV
        for model_name in ['resnet18', 'custom_cnn']:
            print(f"   🔄 Entrenando {model_name}...")
            
            classifier = AlzheimerClassifier(
                num_classes=len(label_names),
                model_name=model_name,
                pretrained=True
            )
            
            model = classifier.build_model()
            classifier.setup_training(train_fold_loader, val_fold_loader)
            
            evaluator = ModelEvaluator(label_names)
            
            # Entrenamiento rápido (menos épocas para CV)
            classifier.train(epochs=10, evaluator=evaluator, save_best=False)
            
            # Evaluar
            val_metrics, _ = evaluator.evaluate_model(
                model, val_fold_loader, classifier.criterion, classifier.device, "validation"
            )
            
            cv_results[model_name]['accuracy'].append(val_metrics['accuracy'])
            cv_results[model_name]['f1_macro'].append(val_metrics['f1_macro'])
            cv_results[model_name]['f1_weighted'].append(val_metrics['f1_weighted'])
            
            print(f"      Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"      F1-macro: {val_metrics['f1_macro']:.4f}")
    
    # Analizar resultados de CV
    print(f"\n📊 RESULTADOS DE VALIDACIÓN CRUZADA")
    print("="*50)
    
    cv_summary = {}
    for model_name, metrics in cv_results.items():
        print(f"\n🏷️  {model_name.upper()}")
        print("-" * 20)
        
        model_summary = {}
        for metric_name, values in metrics.items():
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric_name}: {mean_val:.4f} ± {std_val:.4f}")
            model_summary[metric_name] = {'mean': mean_val, 'std': std_val, 'values': values}
        
        cv_summary[model_name] = model_summary
    
    # Visualizar resultados de CV
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics_names = ['accuracy', 'f1_macro', 'f1_weighted']
    
    for i, metric in enumerate(metrics_names):
        data_to_plot = []
        labels = []
        
        for model_name in cv_results.keys():
            data_to_plot.append(cv_results[model_name][metric])
            labels.append(model_name)
        
        axes[i].boxplot(data_to_plot, labels=labels)
        axes[i].set_title(f'{metric.replace("_", "-").title()}')
        axes[i].set_ylabel('Score')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('figures/cross_validation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return cv_summary

def generate_final_report(models_results, cv_results, label_names):
    """Genera el reporte final del estudio"""
    print(f"\n📄 GENERANDO REPORTE FINAL")
    print("="*40)
    
    # Crear reporte en markdown
    report_md = f"""# Estudio de Detección de Alzheimer mediante Deep Learning

## Resumen Ejecutivo

Este estudio evalúa diferentes arquitecturas de deep learning para la clasificación automática de tomografías cerebrales con el objetivo de detectar diferentes niveles de Alzheimer.

### Dataset
- **Fuente**: Falah/Alzheimer_MRI (HuggingFace)
- **Clases**: {', '.join(label_names)}
- **Número de clases**: {len(label_names)}

## Metodología

### Preprocesamiento
- Conversión a escala de grises → 3 canales (RGB)
- Redimensionado a 224×224 píxeles
- Normalización tipo ImageNet
- Aumentaciones de datos: rotación (±10°), traslación (±5%), ajuste de brillo/contraste (±10%)
- Ruido gaussiano suave (σ=0.02)

### Modelos Evaluados
"""
    
    for model_name in models_results.keys():
        report_md += f"- **{model_name}**: "
        if 'resnet' in model_name:
            report_md += "Red residual preentrenada en ImageNet\n"
        elif 'efficientnet' in model_name:
            report_md += "EfficientNet preentrenada en ImageNet\n"
        elif 'custom' in model_name:
            report_md += "CNN personalizada entrenada desde cero\n"
        else:
            report_md += "Arquitectura personalizada\n"
    
    report_md += "\n## Resultados\n\n### Métricas de Test\n\n"
    
    # Tabla de resultados
    report_md += "| Modelo | Accuracy | F1-Macro | F1-Weighted | Precision-Macro | Recall-Macro | AUC-Macro |\n"
    report_md += "|--------|----------|----------|-------------|-----------------|--------------|----------|\n"
    
    for model_name, results in models_results.items():
        metrics = results['test_metrics']
        auc = metrics.get('auc_macro', 'N/A')
        if isinstance(auc, float):
            auc = f"{auc:.4f}"
        
        report_md += f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['f1_macro']:.4f} | {metrics['f1_weighted']:.4f} | {metrics['precision_macro']:.4f} | {metrics['recall_macro']:.4f} | {auc} |\n"
    
    # Validación cruzada
    if cv_results:
        report_md += "\n### Validación Cruzada (5-fold)\n\n"
        report_md += "| Modelo | Accuracy (μ±σ) | F1-Macro (μ±σ) | F1-Weighted (μ±σ) |\n"
        report_md += "|--------|----------------|----------------|------------------|\n"
        
        for model_name, cv_metrics in cv_results.items():
            acc_mean = cv_metrics['accuracy']['mean']
            acc_std = cv_metrics['accuracy']['std']
            f1_mean = cv_metrics['f1_macro']['mean']
            f1_std = cv_metrics['f1_macro']['std']
            f1w_mean = cv_metrics['f1_weighted']['mean']
            f1w_std = cv_metrics['f1_weighted']['std']
            
            report_md += f"| {model_name} | {acc_mean:.4f}±{acc_std:.4f} | {f1_mean:.4f}±{f1_std:.4f} | {f1w_mean:.4f}±{f1w_std:.4f} |\n"
    
    # Conclusiones
    best_model = max(models_results.items(), key=lambda x: x[1]['test_metrics']['f1_macro'])
    best_model_name, best_results = best_model
    
    report_md += f"\n## Conclusiones\n\n"
    report_md += f"- **Mejor modelo**: {best_model_name} con F1-macro de {best_results['test_metrics']['f1_macro']:.4f}\n"
    report_md += f"- **Accuracy del mejor modelo**: {best_results['test_metrics']['accuracy']:.4f}\n"
    report_md += f"- **Tiempo de entrenamiento**: {best_results['training_time']/60:.1f} minutos\n"
    
    report_md += "\n### Recomendaciones\n\n"
    if best_results['test_metrics']['f1_macro'] > 0.8:
        report_md += "- El modelo muestra un rendimiento excelente para aplicaciones clínicas.\n"
    elif best_results['test_metrics']['f1_macro'] > 0.7:
        report_md += "- El modelo muestra un rendimiento bueno, pero podría beneficiarse de más datos o ajustes.\n"
    else:
        report_md += "- El modelo necesita mejoras significativas antes de uso clínico.\n"
    
    report_md += "- Se recomienda validación adicional con datos de múltiples centros médicos.\n"
    report_md += "- Considerar técnicas de interpretabilidad (Grad-CAM, LIME) para explicar las predicciones.\n"
    
    report_md += f"\n## Archivos Generados\n\n"
    report_md += "### Modelos\n"
    for model_name in models_results.keys():
        report_md += f"- `models/best_model_{model_name}.pt`: Mejor modelo entrenado\n"
    
    report_md += "\n### Visualizaciones\n"
    for model_name in models_results.keys():
        report_md += f"- `figures/confusion_matrix_{model_name}.png`: Matriz de confusión\n"
        report_md += f"- `figures/roc_curves_{model_name}.png`: Curvas ROC\n"
        report_md += f"- `figures/training_history_{model_name}.png`: Historial de entrenamiento\n"
    
    report_md += "- `figures/models_comparison.png`: Comparación entre modelos\n"
    report_md += "- `figures/cross_validation_results.png`: Resultados de validación cruzada\n"
    
    report_md += "\n### Datos\n"
    report_md += "- `results/alzheimer_study_results.json`: Resultados completos en formato JSON\n"
    
    # Guardar reporte
    with open('results/REPORTE_FINAL.md', 'w', encoding='utf-8') as f:
        f.write(report_md)
    
    print("✅ Reporte final generado: results/REPORTE_FINAL.md")
    
    return report_md

def main():
    """Función principal que ejecuta el estudio completo"""
    parser = argparse.ArgumentParser(description='Estudio completo de detección de Alzheimer')
    parser.add_argument('--epochs', type=int, default=20, help='Número de épocas de entrenamiento')
    parser.add_argument('--batch-size', type=int, default=32, help='Tamaño del batch')
    parser.add_argument('--models', nargs='+', default=['resnet18', 'resnet50', 'custom_cnn'], 
                       help='Modelos a entrenar')
    parser.add_argument('--skip-analysis', action='store_true', help='Saltar análisis exploratorio')
    parser.add_argument('--skip-cv', action='store_true', help='Saltar validación cruzada')
    parser.add_argument('--quick-test', action='store_true', help='Ejecutar prueba rápida (menos épocas)')
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 3
        args.models = ['resnet18']
        print("🚀 Modo prueba rápida activado")
    
    print("🧠 ESTUDIO DE DETECCIÓN DE ALZHEIMER")
    print("="*60)
    print(f"📅 Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏗️  Modelos a entrenar: {', '.join(args.models)}")
    print(f"⏱️  Épocas por modelo: {args.epochs}")
    print(f"📦 Tamaño de batch: {args.batch_size}")
    
    # Configurar entorno
    device = setup_environment()
    
    # 1. Análisis exploratorio (opcional)
    if not args.skip_analysis:
        print(f"\n{'='*60}")
        print("FASE 1: ANÁLISIS EXPLORATORIO")
        print("="*60)
        
        analyzer = AlzheimerDatasetAnalyzer()
        analyzer.load_dataset()
        analyzer.create_metadata_df()
        analyzer.exploratory_analysis()
        analyzer.quality_analysis()
        analyzer.show_sample_images()
        
        # Mover figuras a carpeta figures
        for fig_file in ['exploratory_analysis.png', 'quality_analysis.png', 
                        'sample_images.png', 'augmentation_examples.png']:
            if os.path.exists(fig_file):
                os.rename(fig_file, f'figures/{fig_file}')
    else:
        analyzer = AlzheimerDatasetAnalyzer()
        analyzer.load_dataset()
    
    # 2. Preprocesamiento
    print(f"\n{'='*60}")
    print("FASE 2: PREPROCESAMIENTO")
    print("="*60)
    
    preprocessor = DataPreprocessor()
    data_splits = preprocessor.create_data_splits(analyzer.dataset)
    train_idx, val_idx, test_idx, hf_train, hf_test = data_splits
    
    # Mostrar aumentaciones
    preprocessor.visualize_augmentations(analyzer.dataset)
    if os.path.exists('augmentation_examples.png'):
        os.rename('augmentation_examples.png', 'figures/augmentation_examples.png')
    
    label_names = analyzer.label_names
    
    # Calcular pesos de clase
    class_weights = calculate_class_weights(hf_train, train_idx, label_names)
    
    # Crear data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        preprocessor, data_splits, batch_size=args.batch_size
    )
    
    # 3. Entrenamiento y evaluación de modelos
    print(f"\n{'='*60}")
    print("FASE 3: ENTRENAMIENTO Y EVALUACIÓN")
    print("="*60)
    
    models_results = {}
    
    for model_name in args.models:
        try:
            results = train_and_evaluate_model(
                model_name=model_name,
                label_names=label_names,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                class_weights=class_weights,
                epochs=args.epochs,
                pretrained=True
            )
            models_results[model_name] = results
            
        except Exception as e:
            print(f"❌ Error entrenando {model_name}: {str(e)}")
            continue
    
    # 4. Validación cruzada (opcional)
    cv_results = None
    if not args.skip_cv and len(models_results) > 0:
        print(f"\n{'='*60}")
        print("FASE 4: VALIDACIÓN CRUZADA")
        print("="*60)
        
        try:
            cv_results = cross_validation_analysis(
                preprocessor, data_splits, label_names, k_folds=5
            )
        except Exception as e:
            print(f"⚠️  Error en validación cruzada: {str(e)}")
    
    # 5. Comparación y reporte final
    if len(models_results) > 1:
        print(f"\n{'='*60}")
        print("FASE 5: COMPARACIÓN DE MODELOS")
        print("="*60)
        
        comparison_df = compare_models(models_results, label_names)
        comparison_df.to_csv('results/models_comparison.csv', index=False)
        
        if os.path.exists('models_comparison.png'):
            os.rename('models_comparison.png', 'figures/models_comparison.png')
    
    # 6. Guardar resultados y generar reporte
    print(f"\n{'='*60}")
    print("FASE 6: REPORTE FINAL")
    print("="*60)
    
    # Guardar resultados en JSON
    save_results_report(models_results, label_names, 'results/alzheimer_study_results.json')
    
    # Generar reporte final
    final_report = generate_final_report(models_results, cv_results, label_names)
    
    # Resumen final
    print(f"\n🎉 ESTUDIO COMPLETADO")
    print("="*30)
    print(f"📊 Modelos entrenados: {len(models_results)}")
    
    if models_results:
        best_model = max(models_results.items(), key=lambda x: x[1]['test_metrics']['f1_macro'])
        best_model_name, best_results = best_model
        print(f"🏆 Mejor modelo: {best_model_name}")
        print(f"📈 Mejor F1-macro: {best_results['test_metrics']['f1_macro']:.4f}")
        print(f"🎯 Accuracy: {best_results['test_metrics']['accuracy']:.4f}")
    
    print(f"\n📁 Archivos generados en:")
    print(f"   - results/: Reportes y datos")
    print(f"   - models/: Modelos entrenados")
    print(f"   - figures/: Visualizaciones")
    
    print(f"\n📖 Ver reporte completo en: results/REPORTE_FINAL.md")

if __name__ == "__main__":
    main()