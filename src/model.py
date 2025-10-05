"""
Definición del modelo y utilidades de entrenamiento
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix
from tqdm import tqdm
import time

try:
    from .config import *
except ImportError:
    from config import *


def create_model(num_classes=NUM_CLASSES):
    """Crea el modelo ResNet18 preentrenado"""
    print("Creando modelo ResNet18...")
    
    # Cargar ResNet18 preentrenado
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        print("Usando pesos IMAGENET1K_V1")
    except Exception:
        model = models.resnet18(pretrained=True)
        print("Usando pretrained=True (versión anterior)")
    
    # Modificar la capa final para nuestras clases
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    
    # Mover a dispositivo
    model = model.to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modelo creado en {DEVICE}")
    print(f"Parámetros entrenables: {num_params:,}")
    
    return model


def calculate_class_weights(hf_train, train_idx, label_names):
    """Calcula pesos por clase para manejar desbalance"""
    y_train = pd.Series([hf_train[i]["label"] for i in train_idx])
    counts = y_train.value_counts().sort_index()
    
    # Asegurar que tenemos todas las clases
    all_class_counts = []
    for class_id in range(len(label_names)):
        if class_id in counts.index:
            all_class_counts.append(counts[class_id])
        else:
            all_class_counts.append(1)  # Valor mínimo para clases ausentes
    
    # Pesos inversamente proporcionales a la frecuencia
    class_weights = 1.0 / torch.tensor(all_class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(label_names)
    
    print("Pesos por clase:")
    for i, (name, weight) in enumerate(zip(label_names, class_weights.tolist())):
        print(f"  {name}: {weight:.3f}")
    
    return class_weights


def create_optimizer_scheduler(model, class_weights):
    """Crea optimizador, criterio de pérdida y scheduler"""
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3, 
            verbose=True
        )
    except TypeError:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max', 
            factor=0.5, 
            patience=3
        )
    
    return criterion, optimizer, scheduler


def train_one_epoch(model, loader, optimizer, criterion, device, use_amp=True):
    """Entrena una época"""
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and torch.cuda.is_available())
    
    pbar = tqdm(loader, desc="Training")
    for batch_idx, (data, target, _) in enumerate(pbar):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass con mixed precision si está disponible
        with torch.cuda.amp.autocast(enabled=use_amp and torch.cuda.is_available()):
            output = model(data)
            loss = criterion(output, target)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * data.size(0)
        num_samples += data.size(0)
        
        # Actualizar barra de progreso
        pbar.set_postfix({'loss': total_loss / num_samples})
    
    return total_loss / num_samples


@torch.no_grad()
def evaluate_model(model, loader, criterion, device, label_names):
    """Evalúa el modelo y retorna métricas"""
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    pbar = tqdm(loader, desc="Evaluating")
    for data, target, _ in pbar:
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        
        output = model(data)
        loss = criterion(output, target)
        
        total_loss += loss.item() * data.size(0)
        num_samples += data.size(0)
        
        # Obtener predicciones y probabilidades
        probabilities = torch.softmax(output, dim=1)
        predictions = probabilities.argmax(dim=1)
        
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())
    
    # Concatenar todos los resultados
    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    y_prob = np.concatenate(all_probabilities)
    
    # Calcular métricas
    avg_loss = total_loss / num_samples
    accuracy = (y_true == y_pred).mean()
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    try:
        auc_macro = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
    except Exception:
        auc_macro = np.nan
    
    # Reporte detallado
    try:
        report = classification_report(
            y_true, y_pred, 
            target_names=label_names, 
            digits=3, 
            output_dict=True,
            labels=list(range(len(label_names)))
        )
    except ValueError:
        # Si no tenemos todas las clases, usar solo las presentes
        unique_labels = sorted(list(set(y_true) | set(y_pred)))
        present_label_names = [label_names[i] for i in unique_labels]
        report = classification_report(
            y_true, y_pred, 
            target_names=present_label_names, 
            digits=3, 
            output_dict=True,
            labels=unique_labels
        )
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'auc_macro': auc_macro,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                label_names, epochs=EPOCHS, save_path=MODEL_SAVE_PATH):
    """Función principal de entrenamiento"""
    print(f"Iniciando entrenamiento por {epochs} épocas...")
    print(f"Dispositivo: {DEVICE}")
    
    best_f1 = -1.0
    best_epoch = -1
    training_history = []
    
    # Crear directorio para guardar modelo
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'='*50}")
        print(f"Época {epoch}/{epochs}")
        print(f"{'='*50}")
        
        # Entrenamiento
        start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        train_time = time.time() - start_time
        
        # Validación
        start_time = time.time()
        val_metrics = evaluate_model(model, val_loader, criterion, DEVICE, label_names)
        val_time = time.time() - start_time
        
        # Actualizar scheduler
        try:
            scheduler.step(val_metrics['f1_macro'])
        except Exception:
            pass
        
        # Obtener learning rate actual
        current_lr = optimizer.param_groups[0]['lr']
        
        # Mostrar resultados
        print(f"\nResultados Época {epoch}:")
        print(f"  Train Loss: {train_loss:.4f} (tiempo: {train_time:.1f}s)")
        print(f"  Val Loss: {val_metrics['loss']:.4f} (tiempo: {val_time:.1f}s)")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.3f}")
        print(f"  Val F1-Macro: {val_metrics['f1_macro']:.3f}")
        print(f"  Val AUC-Macro: {val_metrics['auc_macro']:.3f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Guardar mejor modelo
        if val_metrics['f1_macro'] > best_f1:
            best_f1 = val_metrics['f1_macro']
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Nuevo mejor modelo guardado (F1: {best_f1:.3f})")
        
        # Guardar historial
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_f1_macro': val_metrics['f1_macro'],
            'val_auc_macro': val_metrics['auc_macro'],
            'learning_rate': current_lr,
            'train_time': train_time,
            'val_time': val_time
        })
    
    print(f"\n{'='*50}")
    print(f"Entrenamiento completado!")
    print(f"Mejor F1-Macro: {best_f1:.3f} (época {best_epoch})")
    print(f"Modelo guardado en: {save_path}")
    print(f"{'='*50}")
    
    return training_history, best_f1, best_epoch