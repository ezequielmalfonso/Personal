# Estudio de Detección de Alzheimer mediante Deep Learning
# Análisis de tomografías cerebrales para clasificación de niveles de Alzheimer

import os, re, random, math, time
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import (f1_score, roc_auc_score, classification_report, 
                           confusion_matrix, roc_curve, auc, precision_recall_curve)
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

# Configuración global
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("="*60)
print("ESTUDIO DE DETECCIÓN DE ALZHEIMER")
print("="*60)
print(f"PyTorch: {torch.__version__}")
print(f"Torchvision: {torchvision.__version__}")
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

class AlzheimerDatasetAnalyzer:
    """Clase para análisis exploratorio del dataset de Alzheimer"""
    
    def __init__(self, dataset_name="Falah/Alzheimer_MRI"):
        self.dataset_name = dataset_name
        self.dataset = None
        self.df_metadata = None
        self.label_names = None
        
    def load_dataset(self):
        """Carga el dataset desde HuggingFace"""
        print("\n📊 Cargando dataset...")
        self.dataset = load_dataset(self.dataset_name)
        
        # Recodificar labels si es necesario (convertir clase 1 a 0)
        def relabel(example):
            if example["label"] == 1:
                example["label"] = 0
            return example
        
        self.dataset['train'] = self.dataset['train'].map(relabel)
        self.dataset['test'] = self.dataset['test'].map(relabel)
        
        self.label_names = self.dataset['train'].features['label'].names
        print(f"✅ Dataset cargado: {len(self.dataset['train'])} train, {len(self.dataset['test'])} test")
        print(f"📋 Clases: {self.label_names}")
        
    def create_metadata_df(self):
        """Crea DataFrame con metadatos de las imágenes"""
        print("\n🔍 Extrayendo metadatos...")
        rows = []
        
        for split_name in ['train', 'test']:
            split_data = self.dataset[split_name]
            for i in tqdm(range(len(split_data)), desc=f"Procesando {split_name}"):
                sample = split_data[i]
                img = sample["image"]
                w, h = img.size
                lbl_id = sample["label"]
                lbl_name = self.label_names[lbl_id]
                
                rows.append({
                    'split': split_name,
                    'index': i,
                    'label_id': lbl_id,
                    'label_name': lbl_name,
                    'width': w,
                    'height': h,
                    'aspect_ratio': w/h,
                    'total_pixels': w*h
                })
        
        self.df_metadata = pd.DataFrame(rows)
        print(f"✅ Metadatos extraídos: {len(self.df_metadata)} imágenes")
        
    def exploratory_analysis(self):
        """Realiza análisis exploratorio completo"""
        print("\n📈 ANÁLISIS EXPLORATORIO")
        print("="*40)
        
        # Distribución por clase y split
        class_dist = self.df_metadata.groupby(['split', 'label_name']).size().unstack(fill_value=0)
        print("\n📊 Distribución por clase:")
        print(class_dist)
        
        # Porcentajes
        pct_dist = class_dist.div(class_dist.sum(axis=1), axis=0) * 100
        print("\n📊 Distribución porcentual:")
        print(pct_dist.round(2))
        
        # Estadísticas de dimensiones
        print("\n📏 Estadísticas de dimensiones:")
        dim_stats = self.df_metadata[['width', 'height', 'aspect_ratio', 'total_pixels']].describe()
        print(dim_stats)
        
        # Visualizaciones
        self._create_exploratory_plots()
        
    def _create_exploratory_plots(self):
        """Crea visualizaciones para el análisis exploratorio"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Distribución por clase (train)
        train_data = self.df_metadata[self.df_metadata['split'] == 'train']
        class_counts = train_data['label_name'].value_counts()
        axes[0,0].bar(class_counts.index, class_counts.values)
        axes[0,0].set_title('Distribución de Clases (Train)')
        axes[0,0].set_xlabel('Clase')
        axes[0,0].set_ylabel('Cantidad')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Distribución de dimensiones
        axes[0,1].hist(self.df_metadata['width'], bins=30, alpha=0.7, label='Width')
        axes[0,1].hist(self.df_metadata['height'], bins=30, alpha=0.7, label='Height')
        axes[0,1].set_title('Distribución de Dimensiones')
        axes[0,1].set_xlabel('Píxeles')
        axes[0,1].set_ylabel('Frecuencia')
        axes[0,1].legend()
        
        # 3. Aspect ratio por clase
        sns.boxplot(data=self.df_metadata, x='label_name', y='aspect_ratio', ax=axes[0,2])
        axes[0,2].set_title('Aspect Ratio por Clase')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. Comparación train vs test
        split_comparison = self.df_metadata.groupby(['split', 'label_name']).size().unstack()
        split_comparison.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Comparación Train vs Test')
        axes[1,0].set_xlabel('Split')
        axes[1,0].set_ylabel('Cantidad')
        axes[1,0].tick_params(axis='x', rotation=0)
        
        # 5. Distribución de píxeles totales
        axes[1,1].hist(self.df_metadata['total_pixels'], bins=30)
        axes[1,1].set_title('Distribución de Píxeles Totales')
        axes[1,1].set_xlabel('Píxeles Totales')
        axes[1,1].set_ylabel('Frecuencia')
        
        # 6. Heatmap de correlaciones
        numeric_cols = ['width', 'height', 'aspect_ratio', 'total_pixels', 'label_id']
        corr_matrix = self.df_metadata[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,2])
        axes[1,2].set_title('Matriz de Correlación')
        
        plt.tight_layout()
        plt.savefig('exploratory_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def quality_analysis(self):
        """Análisis de calidad de las imágenes"""
        print("\n🔍 ANÁLISIS DE CALIDAD")
        print("="*30)
        
        def to_gray_float01(pil_img):
            """Convierte PIL a escala de grises y normaliza a [0,1]"""
            g = pil_img.convert("L")
            arr = np.asarray(g, dtype=np.float32)
            return arr / 255.0
        
        stats = []
        corrupt_count = 0
        low_var_count = 0
        
        for split_name in ['train', 'test']:
            split_data = self.dataset[split_name]
            for i in tqdm(range(len(split_data)), desc=f"Analizando calidad {split_name}"):
                try:
                    pil_img = split_data[i]["image"]
                    arr = to_gray_float01(pil_img)
                    
                    mu = float(arr.mean())
                    sd = float(arr.std())
                    var = float(sd * sd)
                    pct_black = float((arr < 0.02).mean())
                    pct_white = float((arr > 0.98).mean())
                    
                    if sd < 0.02:
                        low_var_count += 1
                    
                    stats.append({
                        'split': split_name,
                        'index': i,
                        'mean_intensity': mu,
                        'std_intensity': sd,
                        'variance': var,
                        'pct_black': pct_black,
                        'pct_white': pct_white
                    })
                    
                except Exception:
                    corrupt_count += 1
        
        self.df_quality = pd.DataFrame(stats)
        
        print(f"📊 Imágenes analizadas: {len(self.df_quality)}")
        print(f"⚠️  Imágenes corruptas: {corrupt_count}")
        print(f"⚠️  Imágenes baja varianza: {low_var_count}")
        
        print("\n📈 Estadísticas de calidad:")
        quality_stats = self.df_quality[['mean_intensity', 'std_intensity', 'pct_black', 'pct_white']].describe()
        print(quality_stats)
        
        # Visualización de calidad
        self._plot_quality_analysis()
        
    def _plot_quality_analysis(self):
        """Visualiza el análisis de calidad"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Distribución de intensidad media
        axes[0,0].hist(self.df_quality['mean_intensity'], bins=50)
        axes[0,0].set_title('Distribución de Intensidad Media')
        axes[0,0].set_xlabel('Intensidad Media')
        axes[0,0].set_ylabel('Frecuencia')
        
        # Distribución de desviación estándar
        axes[0,1].hist(self.df_quality['std_intensity'], bins=50)
        axes[0,1].set_title('Distribución de Desviación Estándar')
        axes[0,1].set_xlabel('Desviación Estándar')
        axes[0,1].set_ylabel('Frecuencia')
        
        # Porcentaje de píxeles negros
        axes[1,0].hist(self.df_quality['pct_black'], bins=50)
        axes[1,0].set_title('Distribución de Píxeles Negros')
        axes[1,0].set_xlabel('% Píxeles Negros')
        axes[1,0].set_ylabel('Frecuencia')
        
        # Scatter: media vs std
        axes[1,1].scatter(self.df_quality['mean_intensity'], self.df_quality['std_intensity'], alpha=0.6)
        axes[1,1].set_title('Intensidad Media vs Desviación Estándar')
        axes[1,1].set_xlabel('Intensidad Media')
        axes[1,1].set_ylabel('Desviación Estándar')
        
        plt.tight_layout()
        plt.savefig('quality_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def show_sample_images(self, n_samples=12):
        """Muestra imágenes de muestra de cada clase"""
        print(f"\n🖼️  MUESTRAS DE IMÁGENES ({n_samples} por clase)")
        print("="*50)
        
        samples_per_class = n_samples // len(self.label_names)
        
        fig, axes = plt.subplots(len(self.label_names), samples_per_class, 
                                figsize=(3*samples_per_class, 3*len(self.label_names)))
        
        if len(self.label_names) == 1:
            axes = axes.reshape(1, -1)
        
        train_data = self.dataset['train']
        
        for class_idx, class_name in enumerate(self.label_names):
            # Encontrar muestras de esta clase
            class_samples = [i for i in range(len(train_data)) 
                           if train_data[i]['label'] == class_idx]
            
            # Seleccionar muestras aleatorias
            selected_samples = random.sample(class_samples, 
                                           min(samples_per_class, len(class_samples)))
            
            for sample_idx, data_idx in enumerate(selected_samples):
                img = train_data[data_idx]['image']
                
                if samples_per_class == 1:
                    ax = axes[class_idx]
                else:
                    ax = axes[class_idx, sample_idx]
                    
                ax.imshow(img, cmap='gray')
                ax.set_title(f'{class_name}\n{img.size}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()

class DataPreprocessor:
    """Clase para preprocesamiento de datos"""
    
    def __init__(self, use_gaussian_noise=True, gauss_std=0.02):
        self.use_gaussian_noise = use_gaussian_noise
        self.gauss_std = gauss_std
        self._setup_transforms()
        
    def _setup_transforms(self):
        """Configura las transformaciones"""
        # Transformaciones básicas
        self.to_3ch = transforms.Grayscale(num_output_channels=3)
        self.resize_224 = transforms.Resize((224, 224), InterpolationMode.BILINEAR)
        self.to_tensor = transforms.ToTensor()
        self.imagenet_norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Aumentaciones para entrenamiento
        self.train_augmentations = transforms.Compose([
            transforms.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), 
                                  interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.1, contrast=0.1)
        ])
        
        # Ruido gaussiano
        self.gaussian_noise = self.AddGaussianNoise(self.gauss_std) if self.use_gaussian_noise else transforms.Lambda(lambda x: x)
        
        # Pipelines finales
        self.train_pipeline = transforms.Compose([
            self.to_3ch,
            self.resize_224,
            self.train_augmentations,
            self.to_tensor,
            self.gaussian_noise,
            self.imagenet_norm
        ])
        
        self.eval_pipeline = transforms.Compose([
            self.to_3ch,
            self.resize_224,
            self.to_tensor,
            self.imagenet_norm
        ])
        
    class AddGaussianNoise:
        def __init__(self, std=0.02):
            self.std = float(std)
            
        def __call__(self, x):
            if self.std <= 0:
                return x
            return torch.clamp(x + torch.randn_like(x) * self.std, 0.0, 1.0)
            
        def __repr__(self):
            return f"{self.__class__.__name__}(std={self.std})"
    
    def create_data_splits(self, dataset, val_size=0.15):
        """Crea splits estratificados de los datos"""
        print(f"\n📊 Creando splits de datos (val_size={val_size})")
        
        hf_train = dataset["train"]
        hf_test = dataset["test"]
        
        # Extraer labels para estratificación
        y_train_all = np.array([hf_train[i]["label"] for i in range(len(hf_train))])
        
        # Split estratificado
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=SEED)
        train_idx, val_idx = next(sss.split(np.arange(len(hf_train)), y_train_all))
        
        # Convertir a listas de enteros
        train_idx = [int(i) for i in train_idx.tolist()]
        val_idx = [int(i) for i in val_idx.tolist()]
        test_idx = [int(i) for i in range(len(hf_test))]
        
        print(f"✅ Splits creados - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        return train_idx, val_idx, test_idx, hf_train, hf_test
    
    def visualize_augmentations(self, dataset, sample_idx=0):
        """Visualiza el efecto de las aumentaciones"""
        print("\n🖼️  VISUALIZACIÓN DE AUMENTACIONES")
        print("="*40)
        
        sample = dataset['train'][sample_idx]
        pil_img = sample['image']
        label_name = dataset['train'].features['label'].names[sample['label']]
        
        # Función para desnormalizar
        def denorm_imgnet(t):
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=t.dtype, device=t.device).view(3,1,1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=t.dtype, device=t.device).view(3,1,1)
            x = t * std + mean
            return torch.clamp(x, 0, 1)
        
        # Generar imágenes
        imgs = []
        titles = []
        
        # Original
        orig_vis = transforms.Compose([self.to_3ch, self.resize_224, self.to_tensor])(pil_img)
        imgs.append(orig_vis)
        titles.append("Original")
        
        # Versiones aumentadas
        for i in range(6):
            x = self.train_pipeline(pil_img)
            x_vis = denorm_imgnet(x)
            imgs.append(x_vis)
            titles.append(f"Aumentada {i+1}")
        
        # Versión eval
        x_eval = self.eval_pipeline(pil_img)
        x_eval_vis = denorm_imgnet(x_eval)
        imgs.append(x_eval_vis)
        titles.append("Eval (sin aumentos)")
        
        # Mostrar
        n = len(imgs)
        cols = 4
        rows = int(np.ceil(n/cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes
        
        for i, (img, title) in enumerate(zip(imgs, titles)):
            if i < len(axes):
                axes[i].imshow(np.moveaxis(img.numpy(), 0, 2))
                axes[i].set_title(title, fontsize=10)
                axes[i].axis('off')
        
        # Ocultar ejes no usados
        for i in range(len(imgs), len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"Clase: {label_name}", y=1.02, fontsize=14)
        plt.tight_layout()
        plt.savefig('augmentation_examples.png', dpi=300, bbox_inches='tight')
        plt.show()

class HFDataset(Dataset):
    """Dataset personalizado para datos de HuggingFace"""
    
    def __init__(self, hf_split, indices, transform):
        self.hf = hf_split
        self.idx = [int(i) for i in indices]
        self.tfm = transform
        
    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self, k):
        i = self.idx[k]
        s = self.hf[i]
        x = self.tfm(s['image'])
        y = int(s['label'])
        return x, y, i

def main():
    """Función principal para ejecutar el análisis"""
    print("🚀 Iniciando análisis de detección de Alzheimer...")
    
    # 1. Análisis exploratorio
    analyzer = AlzheimerDatasetAnalyzer()
    analyzer.load_dataset()
    analyzer.create_metadata_df()
    analyzer.exploratory_analysis()
    analyzer.quality_analysis()
    analyzer.show_sample_images()
    
    # 2. Preprocesamiento
    preprocessor = DataPreprocessor()
    train_idx, val_idx, test_idx, hf_train, hf_test = preprocessor.create_data_splits(analyzer.dataset)
    preprocessor.visualize_augmentations(analyzer.dataset)
    
    print("\n✅ Análisis exploratorio completado!")
    print("📁 Archivos generados:")
    print("   - exploratory_analysis.png")
    print("   - quality_analysis.png") 
    print("   - sample_images.png")
    print("   - augmentation_examples.png")
    
    return analyzer, preprocessor, (train_idx, val_idx, test_idx, hf_train, hf_test)

if __name__ == "__main__":
    analyzer, preprocessor, data_splits = main()