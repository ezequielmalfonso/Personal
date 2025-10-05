# Módulo de Modelos y Entrenamiento para Detección de Alzheimer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, auc, precision_recall_curve,
                           f1_score, accuracy_score, precision_score, recall_score)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Clase para evaluación completa de modelos"""
    
    def __init__(self, label_names):
        self.label_names = label_names
        self.num_classes = len(label_names)
        
    def evaluate_model(self, model, data_loader, criterion, device, split_name="test"):
        """Evaluación completa de un modelo"""
        model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_probabilities = []
        all_labels = []
        all_indices = []
        
        with torch.no_grad():
            for batch_idx, (inputs, labels, indices) in enumerate(tqdm(data_loader, desc=f"Evaluando {split_name}")):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                
                total_loss += loss.item() * inputs.size(0)
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_indices.extend(indices.numpy())
        
        # Convertir a arrays numpy
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_prob = np.array(all_probabilities)
        
        # Calcular métricas
        metrics = self._calculate_metrics(y_true, y_pred, y_prob)
        metrics['loss'] = total_loss / len(data_loader.dataset)
        
        # Generar reportes
        metrics['classification_report'] = classification_report(
            y_true, y_pred, target_names=self.label_names, digits=4
        )
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return metrics, (y_true, y_pred, y_prob, all_indices)
    
    def _calculate_metrics(self, y_true, y_pred, y_prob):
        """Calcula métricas de clasificación"""
        metrics = {}
        
        # Métricas básicas
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Métricas por clase
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, class_name in enumerate(self.label_names):
            metrics[f'precision_{class_name}'] = precision_per_class[i]
            metrics[f'recall_{class_name}'] = recall_per_class[i]
            metrics[f'f1_{class_name}'] = f1_per_class[i]
        
        # AUC si es posible
        try:
            if self.num_classes > 2:
                metrics['auc_macro'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                metrics['auc_weighted'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
            else:
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
        except Exception as e:
            print(f"No se pudo calcular AUC: {e}")
            metrics['auc_macro'] = np.nan
            metrics['auc_weighted'] = np.nan
        
        return metrics
    
    def plot_confusion_matrix(self, cm, title="Matriz de Confusión", normalize=False):
        """Visualiza matriz de confusión"""
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_display = cm_norm
            fmt = '.2f'
            cmap = 'Blues'
        else:
            cm_display = cm
            fmt = 'd'
            cmap = 'Oranges'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_display, annot=True, fmt=fmt, cmap=cmap,
                   xticklabels=self.label_names, yticklabels=self.label_names)
        plt.title(title)
        plt.ylabel('Etiqueta Verdadera')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_roc_curves(self, y_true, y_prob, title="Curvas ROC"):
        """Visualiza curvas ROC por clase"""
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, class_name in enumerate(self.label_names):
            # Crear etiquetas binarias para esta clase
            y_true_binary = (y_true == i).astype(int)
            
            if len(np.unique(y_true_binary)) < 2:
                continue
                
            fpr, tpr, _ = roc_curve(y_true_binary, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            
            color = colors[i % len(colors)]
            plt.plot(fpr, tpr, color=color, lw=2, 
                    label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Línea de Referencia')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
        plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
        plt.title(title)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_precision_recall_curves(self, y_true, y_prob, title="Curvas Precisión-Recall"):
        """Visualiza curvas Precisión-Recall por clase"""
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, class_name in enumerate(self.label_names):
            y_true_binary = (y_true == i).astype(int)
            
            if len(np.unique(y_true_binary)) < 2:
                continue
                
            precision, recall, _ = precision_recall_curve(y_true_binary, y_prob[:, i])
            pr_auc = auc(recall, precision)
            
            color = colors[i % len(colors)]
            plt.plot(recall, precision, color=color, lw=2,
                    label=f'{class_name} (AUC = {pr_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall (Sensibilidad)')
        plt.ylabel('Precisión')
        plt.title(title)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()

class AlzheimerClassifier:
    """Clasificador principal para detección de Alzheimer"""
    
    def __init__(self, num_classes, model_name='resnet18', pretrained=True):
        self.num_classes = num_classes
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_history = defaultdict(list)
        
    def build_model(self):
        """Construye el modelo"""
        print(f"🏗️  Construyendo modelo {self.model_name}...")
        
        if self.model_name == 'resnet18':
            if self.pretrained:
                try:
                    weights = models.ResNet18_Weights.IMAGENET1K_V1
                    self.model = models.resnet18(weights=weights)
                except:
                    self.model = models.resnet18(pretrained=True)
            else:
                self.model = models.resnet18(pretrained=False)
            
            # Reemplazar la capa final
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
            
        elif self.model_name == 'resnet50':
            if self.pretrained:
                try:
                    weights = models.ResNet50_Weights.IMAGENET1K_V1
                    self.model = models.resnet50(weights=weights)
                except:
                    self.model = models.resnet50(pretrained=True)
            else:
                self.model = models.resnet50(pretrained=False)
            
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
            
        elif self.model_name == 'efficientnet_b0':
            if self.pretrained:
                try:
                    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
                    self.model = models.efficientnet_b0(weights=weights)
                except:
                    self.model = models.efficientnet_b0(pretrained=True)
            else:
                self.model = models.efficientnet_b0(pretrained=False)
            
            in_features = self.model.classifier[1].in_features
            self.model.classifier[1] = nn.Linear(in_features, self.num_classes)
            
        elif self.model_name == 'custom_cnn':
            self.model = self._build_custom_cnn()
        
        else:
            raise ValueError(f"Modelo no soportado: {self.model_name}")
        
        self.model = self.model.to(self.device)
        
        # Contar parámetros
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Modelo {self.model_name} construido")
        print(f"📊 Parámetros totales: {total_params:,}")
        print(f"🎯 Parámetros entrenables: {trainable_params:,}")
        
        return self.model
    
    def _build_custom_cnn(self):
        """Construye una CNN personalizada"""
        class CustomCNN(nn.Module):
            def __init__(self, num_classes):
                super(CustomCNN, self).__init__()
                
                # Bloque convolucional 1
                self.conv1 = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2)
                )
                
                # Bloque convolucional 2
                self.conv2 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2)
                )
                
                # Bloque convolucional 3
                self.conv3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2)
                )
                
                # Bloque convolucional 4
                self.conv4 = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2)
                )
                
                # Pooling adaptativo y clasificador
                self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(256, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(128, num_classes)
                )
                
            def forward(self, x):
                x = self.conv1(x)
                x = self.conv2(x)
                x = self.conv3(x)
                x = self.conv4(x)
                x = self.adaptive_pool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x
        
        return CustomCNN(self.num_classes)
    
    def setup_training(self, train_loader, val_loader, class_weights=None):
        """Configura el entrenamiento"""
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Configurar loss con pesos de clase si se proporcionan
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Configurar optimizador
        self.optimizer = optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=1e-4)
        
        # Configurar scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=3, verbose=True
        )
        
        # Configurar scaler para mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        
        print("✅ Configuración de entrenamiento completada")
    
    def train_epoch(self, epoch):
        """Entrena una época"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Época {epoch}")
        
        for batch_idx, (inputs, labels, _) in enumerate(pbar):
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass con mixed precision
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Estadísticas
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Actualizar barra de progreso
            current_loss = running_loss / total_samples
            current_acc = correct_predictions / total_samples
            pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.4f}'
            })
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = correct_predictions / len(self.train_loader.dataset)
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, evaluator):
        """Valida una época"""
        metrics, _ = evaluator.evaluate_model(
            self.model, self.val_loader, self.criterion, self.device, "validation"
        )
        return metrics
    
    def train(self, epochs, evaluator, save_best=True, early_stopping_patience=7):
        """Entrena el modelo"""
        print(f"🚀 Iniciando entrenamiento por {epochs} épocas...")
        
        best_f1 = -1.0
        best_epoch = -1
        patience_counter = 0
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Entrenar
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validar
            val_metrics = self.validate_epoch(evaluator)
            
            # Actualizar scheduler
            self.scheduler.step(val_metrics['f1_macro'])
            
            # Guardar historial
            self.training_history['epoch'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['val_f1_macro'].append(val_metrics['f1_macro'])
            self.training_history['lr'].append(self.optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - epoch_start
            
            # Imprimir métricas
            print(f"Época {epoch:2d}/{epochs} ({epoch_time:.1f}s) | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Val F1: {val_metrics['f1_macro']:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Guardar mejor modelo
            if val_metrics['f1_macro'] > best_f1:
                best_f1 = val_metrics['f1_macro']
                best_epoch = epoch
                patience_counter = 0
                
                if save_best:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_f1': best_f1,
                        'training_history': dict(self.training_history)
                    }, f'best_model_{self.model_name}.pt')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"⏹️  Early stopping en época {epoch} (paciencia: {early_stopping_patience})")
                break
        
        total_time = time.time() - start_time
        print(f"\n✅ Entrenamiento completado en {total_time/60:.1f} minutos")
        print(f"🏆 Mejor F1-macro: {best_f1:.4f} en época {best_epoch}")
        
        return best_f1, best_epoch
    
    def load_best_model(self):
        """Carga el mejor modelo guardado"""
        checkpoint_path = f'best_model_{self.model_name}.pt'
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Mejor modelo cargado desde {checkpoint_path}")
            return True
        except FileNotFoundError:
            print(f"❌ No se encontró {checkpoint_path}")
            return False
    
    def plot_training_history(self):
        """Visualiza el historial de entrenamiento"""
        if not self.training_history['epoch']:
            print("❌ No hay historial de entrenamiento para mostrar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = self.training_history['epoch']
        
        # Loss
        axes[0,0].plot(epochs, self.training_history['train_loss'], 'b-', label='Train Loss')
        axes[0,0].plot(epochs, self.training_history['val_loss'], 'r-', label='Val Loss')
        axes[0,0].set_title('Pérdida durante el Entrenamiento')
        axes[0,0].set_xlabel('Época')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0,1].plot(epochs, self.training_history['train_acc'], 'b-', label='Train Acc')
        axes[0,1].plot(epochs, self.training_history['val_acc'], 'r-', label='Val Acc')
        axes[0,1].set_title('Precisión durante el Entrenamiento')
        axes[0,1].set_xlabel('Época')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # F1 Score
        axes[1,0].plot(epochs, self.training_history['val_f1_macro'], 'g-', label='Val F1-Macro')
        axes[1,0].set_title('F1-Score durante el Entrenamiento')
        axes[1,0].set_xlabel('Época')
        axes[1,0].set_ylabel('F1-Score')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1,1].plot(epochs, self.training_history['lr'], 'orange', label='Learning Rate')
        axes[1,1].set_title('Learning Rate durante el Entrenamiento')
        axes[1,1].set_xlabel('Época')
        axes[1,1].set_ylabel('Learning Rate')
        axes[1,1].set_yscale('log')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'training_history_{self.model_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig

def compare_models(models_results, label_names):
    """Compara resultados de múltiples modelos"""
    print("\n📊 COMPARACIÓN DE MODELOS")
    print("="*50)
    
    # Crear DataFrame de comparación
    comparison_data = []
    
    for model_name, results in models_results.items():
        metrics = results['test_metrics']
        comparison_data.append({
            'Modelo': model_name,
            'Accuracy': metrics['accuracy'],
            'F1-Macro': metrics['f1_macro'],
            'F1-Weighted': metrics['f1_weighted'],
            'Precision-Macro': metrics['precision_macro'],
            'Recall-Macro': metrics['recall_macro'],
            'AUC-Macro': metrics.get('auc_macro', np.nan)
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.round(4)
    
    print(df_comparison.to_string(index=False))
    
    # Visualización
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    metrics_to_plot = ['Accuracy', 'F1-Macro', 'Precision-Macro', 'Recall-Macro']
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i//2, i%2]
        bars = ax.bar(df_comparison['Modelo'], df_comparison[metric])
        ax.set_title(f'Comparación: {metric}')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('models_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_comparison

def save_results_report(models_results, label_names, output_file='alzheimer_study_results.json'):
    """Guarda un reporte completo de resultados"""
    report = {
        'study_info': {
            'dataset': 'Falah/Alzheimer_MRI',
            'classes': label_names,
            'num_classes': len(label_names),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        },
        'models': {}
    }
    
    for model_name, results in models_results.items():
        model_report = {
            'test_metrics': results['test_metrics'],
            'training_time': results.get('training_time', 'N/A'),
            'best_epoch': results.get('best_epoch', 'N/A'),
            'model_params': results.get('model_params', 'N/A')
        }
        
        # Convertir arrays numpy a listas para JSON
        for key, value in model_report['test_metrics'].items():
            if isinstance(value, np.ndarray):
                model_report['test_metrics'][key] = value.tolist()
            elif isinstance(value, np.floating):
                model_report['test_metrics'][key] = float(value)
        
        report['models'][model_name] = model_report
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Reporte guardado en {output_file}")
    return report