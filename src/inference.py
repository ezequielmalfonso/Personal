"""
Utilidades para inferencia con el modelo entrenado
"""
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import os

try:
    from .config import *
    from .model import create_model
    from .data_utils import get_transforms
except ImportError:
    from config import *
    from model import create_model
    from data_utils import get_transforms


class AlzheimerPredictor:
    """Clase para realizar predicciones con el modelo entrenado"""
    
    def __init__(self, model_path=MODEL_SAVE_PATH, label_names=None):
        self.device = DEVICE
        self.model_path = model_path
        self.label_names = label_names or [
            "MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"
        ]
        
        # Cargar modelo
        self.model = self._load_model()
        
        # Obtener transformaciones (solo eval)
        _, self.transform = get_transforms()
        
        print(f"Predictor inicializado en {self.device}")
        print(f"Clases disponibles: {self.label_names}")
    
    def _load_model(self):
        """Carga el modelo entrenado"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modelo no encontrado en: {self.model_path}")
        
        model = create_model(num_classes=len(self.label_names))
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.eval()
        
        print(f"Modelo cargado desde: {self.model_path}")
        return model
    
    def predict_single_image(self, image_path_or_pil, return_probabilities=True):
        """
        Predice la clase de una sola imagen
        
        Args:
            image_path_or_pil: Ruta a la imagen o objeto PIL Image
            return_probabilities: Si retornar las probabilidades por clase
        
        Returns:
            dict con predicción, confianza y probabilidades (opcional)
        """
        # Cargar imagen si es una ruta
        if isinstance(image_path_or_pil, str):
            if not os.path.exists(image_path_or_pil):
                raise FileNotFoundError(f"Imagen no encontrada: {image_path_or_pil}")
            image = Image.open(image_path_or_pil)
        else:
            image = image_path_or_pil
        
        # Asegurar que es PIL Image
        if not isinstance(image, Image.Image):
            raise ValueError("La entrada debe ser una ruta de archivo o PIL Image")
        
        # Aplicar transformaciones
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Realizar predicción
        with torch.no_grad():
            logits = self.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = probabilities.argmax(dim=1).item()
            confidence = probabilities.max().item()
        
        result = {
            'predicted_class': self.label_names[predicted_class],
            'predicted_class_id': predicted_class,
            'confidence': confidence
        }
        
        if return_probabilities:
            result['probabilities'] = {
                class_name: prob.item() 
                for class_name, prob in zip(self.label_names, probabilities[0])
            }
        
        return result
    
    def predict_batch(self, image_paths, batch_size=32):
        """
        Predice un lote de imágenes
        
        Args:
            image_paths: Lista de rutas a imágenes
            batch_size: Tamaño del lote
        
        Returns:
            Lista de diccionarios con predicciones
        """
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            # Cargar y transformar imágenes del lote
            for path in batch_paths:
                try:
                    image = Image.open(path)
                    tensor = self.transform(image)
                    batch_tensors.append(tensor)
                except Exception as e:
                    print(f"Error procesando {path}: {e}")
                    results.append({
                        'image_path': path,
                        'error': str(e)
                    })
                    continue
            
            if not batch_tensors:
                continue
            
            # Crear lote
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Predicción
            with torch.no_grad():
                logits = self.model(batch_tensor)
                probabilities = torch.softmax(logits, dim=1)
                predicted_classes = probabilities.argmax(dim=1)
                confidences = probabilities.max(dim=1)[0]
            
            # Procesar resultados
            for j, path in enumerate(batch_paths):
                if j < len(predicted_classes):  # Verificar que no hubo error
                    pred_class = predicted_classes[j].item()
                    confidence = confidences[j].item()
                    probs = probabilities[j]
                    
                    results.append({
                        'image_path': path,
                        'predicted_class': self.label_names[pred_class],
                        'predicted_class_id': pred_class,
                        'confidence': confidence,
                        'probabilities': {
                            class_name: prob.item() 
                            for class_name, prob in zip(self.label_names, probs)
                        }
                    })
        
        return results
    
    def visualize_prediction(self, image_path_or_pil, figsize=(12, 5)):
        """
        Visualiza una imagen con su predicción
        
        Args:
            image_path_or_pil: Ruta a la imagen o objeto PIL Image
            figsize: Tamaño de la figura
        """
        # Realizar predicción
        result = self.predict_single_image(image_path_or_pil, return_probabilities=True)
        
        # Cargar imagen original
        if isinstance(image_path_or_pil, str):
            original_image = Image.open(image_path_or_pil)
            title_suffix = f" - {os.path.basename(image_path_or_pil)}"
        else:
            original_image = image_path_or_pil
            title_suffix = ""
        
        # Crear visualización
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Mostrar imagen original
        ax1.imshow(original_image, cmap='gray' if original_image.mode == 'L' else None)
        ax1.set_title(f"Imagen Original{title_suffix}")
        ax1.axis('off')
        
        # Mostrar probabilidades
        classes = list(result['probabilities'].keys())
        probs = list(result['probabilities'].values())
        colors = ['red' if i == result['predicted_class_id'] else 'skyblue' 
                 for i in range(len(classes))]
        
        bars = ax2.bar(classes, probs, color=colors, alpha=0.7)
        ax2.set_title(f"Predicción: {result['predicted_class']}\n"
                     f"Confianza: {result['confidence']:.3f}")
        ax2.set_ylabel('Probabilidad')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Agregar valores en las barras
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        return result


def create_demo_predictions(predictor, demo_images_dir, output_dir="results/demo_predictions"):
    """
    Crea predicciones de demostración para imágenes en un directorio
    
    Args:
        predictor: Instancia de AlzheimerPredictor
        demo_images_dir: Directorio con imágenes de demostración
        output_dir: Directorio donde guardar los resultados
    """
    if not os.path.exists(demo_images_dir):
        print(f"Directorio de demostración no encontrado: {demo_images_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Buscar imágenes
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for file in os.listdir(demo_images_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(demo_images_dir, file))
    
    if not image_files:
        print(f"No se encontraron imágenes en: {demo_images_dir}")
        return
    
    print(f"Procesando {len(image_files)} imágenes de demostración...")
    
    # Realizar predicciones
    results = predictor.predict_batch(image_files)
    
    # Guardar resultados en CSV
    import pandas as pd
    
    df_results = []
    for result in results:
        if 'error' not in result:
            row = {
                'imagen': os.path.basename(result['image_path']),
                'prediccion': result['predicted_class'],
                'confianza': result['confidence']
            }
            # Agregar probabilidades por clase
            for class_name, prob in result['probabilities'].items():
                row[f'prob_{class_name}'] = prob
            
            df_results.append(row)
    
    if df_results:
        df = pd.DataFrame(df_results)
        csv_path = os.path.join(output_dir, "predicciones_demo.csv")
        df.to_csv(csv_path, index=False)
        print(f"Resultados guardados en: {csv_path}")
        
        # Mostrar resumen
        print("\nResumen de predicciones:")
        print(df['prediccion'].value_counts())
        print(f"\nConfianza promedio: {df['confianza'].mean():.3f}")
    
    return results


def explain_prediction(predictor, image_path_or_pil, method='gradcam'):
    """
    Genera explicaciones visuales para una predicción (implementación básica)
    
    Args:
        predictor: Instancia de AlzheimerPredictor
        image_path_or_pil: Imagen a explicar
        method: Método de explicación ('gradcam', 'attention')
    
    Note:
        Esta es una implementación básica. Para explicaciones más avanzadas,
        se recomienda usar librerías como captum o grad-cam.
    """
    print("Función de explicación en desarrollo...")
    print("Para implementar explicaciones avanzadas, considere usar:")
    print("- Captum (https://captum.ai/)")
    print("- Grad-CAM")
    print("- LIME")
    
    # Por ahora, solo mostramos la predicción
    result = predictor.visualize_prediction(image_path_or_pil)
    return result