"""
Utilidades para evaluación y visualización de resultados
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import pandas as pd
import torch
import os

try:
    from .config import *
except ImportError:
    from config import *


def plot_confusion_matrix(cm, labels, normalize=False, title="Matriz de Confusión", 
                         save_path=None, figsize=(8, 6)):
    """Grafica la matriz de confusión"""
    plt.figure(figsize=figsize)
    
    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True).clip(min=1)
        fmt = '.2f'
        cmap = 'Blues'
        title += " (Normalizada)"
    else:
        cm_norm = cm.astype('int')
        fmt = 'd'
        cmap = 'Oranges'
    
    sns.heatmap(cm_norm, annot=True, fmt=fmt, cmap=cmap, 
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Proporción' if normalize else 'Cantidad'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicción', fontsize=12)
    plt.ylabel('Verdad', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matriz de confusión guardada en: {save_path}")
    
    plt.show()


def plot_roc_curves(y_true, y_prob, label_names, save_path=None, figsize=(10, 8)):
    """Grafica las curvas ROC para cada clase"""
    plt.figure(figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (class_name, color) in enumerate(zip(label_names, colors)):
        if i >= len(label_names):
            break
            
        # Crear etiquetas binarias para la clase actual
        y_true_binary = (y_true == i).astype(int)
        
        # Verificar que hay ejemplos de ambas clases
        if y_true_binary.sum() == 0 or y_true_binary.sum() == len(y_true_binary):
            print(f"Advertencia: Clase {class_name} no tiene ejemplos suficientes para ROC")
            continue
        
        # Calcular curva ROC
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color, lw=2, 
                label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    # Línea diagonal (clasificador aleatorio)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Clasificador Aleatorio')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=12)
    plt.title('Curvas ROC por Clase', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Curvas ROC guardadas en: {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None, figsize=(15, 5)):
    """Grafica el historial de entrenamiento"""
    df_history = pd.DataFrame(history)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Pérdida
    axes[0].plot(df_history['epoch'], df_history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(df_history['epoch'], df_history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Pérdida')
    axes[0].set_title('Pérdida durante el Entrenamiento')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Métricas de validación
    axes[1].plot(df_history['epoch'], df_history['val_accuracy'], 'g-', label='Accuracy', linewidth=2)
    axes[1].plot(df_history['epoch'], df_history['val_f1_macro'], 'b-', label='F1-Macro', linewidth=2)
    if not df_history['val_auc_macro'].isna().all():
        axes[1].plot(df_history['epoch'], df_history['val_auc_macro'], 'r-', label='AUC-Macro', linewidth=2)
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Métrica')
    axes[1].set_title('Métricas de Validación')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Learning Rate
    axes[2].plot(df_history['epoch'], df_history['learning_rate'], 'purple', linewidth=2)
    axes[2].set_xlabel('Época')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Historial de entrenamiento guardado en: {save_path}")
    
    plt.show()


def generate_classification_report_df(y_true, y_pred, label_names):
    """Genera un DataFrame con el reporte de clasificación"""
    report_dict = classification_report(y_true, y_pred, target_names=label_names, 
                                      digits=3, output_dict=True)
    
    # Convertir a DataFrame
    df_report = pd.DataFrame(report_dict).transpose()
    
    # Reordenar columnas
    cols = ['precision', 'recall', 'f1-score', 'support']
    df_report = df_report[cols]
    
    # Formatear números
    for col in ['precision', 'recall', 'f1-score']:
        df_report[col] = df_report[col].round(3)
    
    df_report['support'] = df_report['support'].astype(int)
    
    return df_report


def plot_class_distribution(class_distributions, save_path=None, figsize=(12, 4)):
    """Grafica la distribución de clases en train/val/test"""
    train_dist, val_dist, test_dist = class_distributions
    
    # Crear DataFrame para facilitar el plotting
    df_dist = pd.DataFrame({
        'Train': train_dist,
        'Validation': val_dist,
        'Test': test_dist
    }).fillna(0)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Train
    axes[0].bar(df_dist.index, df_dist['Train'], color='skyblue', alpha=0.8)
    axes[0].set_title('Distribución - Entrenamiento')
    axes[0].set_ylabel('Cantidad de muestras')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Validation
    axes[1].bar(df_dist.index, df_dist['Validation'], color='lightcoral', alpha=0.8)
    axes[1].set_title('Distribución - Validación')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Test
    axes[2].bar(df_dist.index, df_dist['Test'], color='lightgreen', alpha=0.8)
    axes[2].set_title('Distribución - Test')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Distribución de clases guardada en: {save_path}")
    
    plt.show()


def save_results_summary(test_metrics, training_history, label_names, save_path):
    """Guarda un resumen completo de los resultados"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE RESULTADOS - DETECCIÓN DE ALZHEIMER\n")
        f.write("=" * 50 + "\n\n")
        
        # Información del modelo
        f.write("CONFIGURACIÓN DEL MODELO:\n")
        f.write(f"- Modelo: ResNet18 preentrenado\n")
        f.write(f"- Clases: {label_names}\n")
        f.write(f"- Épocas entrenadas: {len(training_history)}\n")
        f.write(f"- Dispositivo: {DEVICE}\n")
        f.write(f"- Batch size: {BATCH_SIZE}\n")
        f.write(f"- Learning rate inicial: {LEARNING_RATE}\n\n")
        
        # Mejores métricas de entrenamiento
        best_epoch_idx = np.argmax([h['val_f1_macro'] for h in training_history])
        best_epoch = training_history[best_epoch_idx]
        
        f.write("MEJOR ÉPOCA DE ENTRENAMIENTO:\n")
        f.write(f"- Época: {best_epoch['epoch']}\n")
        f.write(f"- Train Loss: {best_epoch['train_loss']:.4f}\n")
        f.write(f"- Val Loss: {best_epoch['val_loss']:.4f}\n")
        f.write(f"- Val Accuracy: {best_epoch['val_accuracy']:.3f}\n")
        f.write(f"- Val F1-Macro: {best_epoch['val_f1_macro']:.3f}\n")
        f.write(f"- Val AUC-Macro: {best_epoch['val_auc_macro']:.3f}\n\n")
        
        # Resultados en test
        f.write("RESULTADOS EN TEST:\n")
        f.write(f"- Test Loss: {test_metrics['loss']:.4f}\n")
        f.write(f"- Test Accuracy: {test_metrics['accuracy']:.3f}\n")
        f.write(f"- Test F1-Macro: {test_metrics['f1_macro']:.3f}\n")
        f.write(f"- Test AUC-Macro: {test_metrics['auc_macro']:.3f}\n\n")
        
        # Reporte detallado por clase
        f.write("REPORTE DETALLADO POR CLASE:\n")
        report_lines = classification_report(
            test_metrics['y_true'], 
            test_metrics['y_pred'], 
            target_names=label_names, 
            digits=3
        ).split('\n')
        
        for line in report_lines:
            f.write(line + '\n')
        
        f.write("\n" + "=" * 50 + "\n")
    
    print(f"Resumen de resultados guardado en: {save_path}")


def create_presentation_plots(test_metrics, training_history, label_names, 
                            class_distributions, results_dir=RESULTS_PATH):
    """Crea todas las visualizaciones para la presentación"""
    os.makedirs(results_dir, exist_ok=True)
    
    print("Generando visualizaciones para presentación...")
    
    # 1. Matriz de confusión (absoluta y normalizada)
    plot_confusion_matrix(
        test_metrics['confusion_matrix'], 
        label_names, 
        normalize=False,
        title="Matriz de Confusión - Valores Absolutos",
        save_path=os.path.join(results_dir, "confusion_matrix_absolute.png")
    )
    
    plot_confusion_matrix(
        test_metrics['confusion_matrix'], 
        label_names, 
        normalize=True,
        title="Matriz de Confusión - Normalizada",
        save_path=os.path.join(results_dir, "confusion_matrix_normalized.png")
    )
    
    # 2. Curvas ROC
    plot_roc_curves(
        test_metrics['y_true'], 
        test_metrics['y_prob'], 
        label_names,
        save_path=os.path.join(results_dir, "roc_curves.png")
    )
    
    # 3. Historial de entrenamiento
    plot_training_history(
        training_history,
        save_path=os.path.join(results_dir, "training_history.png")
    )
    
    # 4. Distribución de clases
    plot_class_distribution(
        class_distributions,
        save_path=os.path.join(results_dir, "class_distribution.png")
    )
    
    # 5. Resumen de resultados
    save_results_summary(
        test_metrics, 
        training_history, 
        label_names,
        save_path=os.path.join(results_dir, "results_summary.txt")
    )
    
    print(f"Todas las visualizaciones guardadas en: {results_dir}")


def analyze_misclassifications(test_metrics, label_names, top_n=5):
    """Analiza las clasificaciones erróneas más comunes"""
    y_true = test_metrics['y_true']
    y_pred = test_metrics['y_pred']
    y_prob = test_metrics['y_prob']
    
    # Encontrar clasificaciones erróneas
    misclassified = y_true != y_pred
    misclassified_indices = np.where(misclassified)[0]
    
    if len(misclassified_indices) == 0:
        print("¡No hay clasificaciones erróneas!")
        return
    
    print(f"Análisis de {len(misclassified_indices)} clasificaciones erróneas:")
    print("-" * 60)
    
    # Confianza en predicciones erróneas
    misclassified_probs = y_prob[misclassified_indices]
    misclassified_confidence = np.max(misclassified_probs, axis=1)
    
    print(f"Confianza promedio en predicciones erróneas: {misclassified_confidence.mean():.3f}")
    print(f"Confianza mínima: {misclassified_confidence.min():.3f}")
    print(f"Confianza máxima: {misclassified_confidence.max():.3f}")
    
    # Pares de confusión más comunes
    confusion_pairs = []
    for i in misclassified_indices:
        true_class = label_names[y_true[i]]
        pred_class = label_names[y_pred[i]]
        confidence = misclassified_confidence[i - misclassified_indices[0]]
        confusion_pairs.append((true_class, pred_class, confidence))
    
    # Contar pares más frecuentes
    from collections import Counter
    pair_counts = Counter([(true, pred) for true, pred, _ in confusion_pairs])
    
    print(f"\nPares de confusión más frecuentes:")
    for (true_class, pred_class), count in pair_counts.most_common(top_n):
        print(f"  {true_class} → {pred_class}: {count} casos")
    
    return misclassified_indices, confusion_pairs