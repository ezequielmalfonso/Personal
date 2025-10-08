"""
Utilidades para manejo de datos y transformaciones
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import StratifiedShuffleSplit
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import os

try:
    from .config import *
except ImportError:
    from config import *


class AddGaussianNoise:
    """Agrega ruido gaussiano suave a las imágenes"""
    def __init__(self, std=GAUSS_STD):
        self.std = float(std)
    
    def __call__(self, x: torch.Tensor):
        if self.std <= 0:
            return x
        return torch.clamp(x + torch.randn_like(x) * self.std, 0.0, 1.0)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(std={self.std})"


class HFDataset(Dataset):
    """Dataset personalizado para Hugging Face datasets"""
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


def get_transforms():
    """Crea las transformaciones para entrenamiento y evaluación"""
    # Transformaciones base
    to_3ch = transforms.Grayscale(num_output_channels=3)
    resize_224 = transforms.Resize((224, 224), InterpolationMode.BILINEAR)
    to_tensor = transforms.ToTensor()
    imagenet_norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    # Augmentaciones para entrenamiento
    rot_aug = transforms.RandomRotation(
        degrees=ROTATION_DEGREES,
        interpolation=InterpolationMode.BILINEAR
    )
    shift_aug = transforms.RandomAffine(
        degrees=0, 
        translate=(TRANSLATE_RATIO, TRANSLATE_RATIO),
        interpolation=InterpolationMode.BILINEAR
    )
    jitter = transforms.ColorJitter(
        brightness=BRIGHTNESS_FACTOR, 
        contrast=CONTRAST_FACTOR
    )
    
    gauss = AddGaussianNoise(std=GAUSS_STD) if USE_GAUSSIAN_NOISE else transforms.Lambda(lambda t: t)
    
    # Pipeline de entrenamiento (con augmentaciones)
    train_pipeline = transforms.Compose([
        to_3ch,
        resize_224,
        rot_aug,
        shift_aug,
        jitter,
        to_tensor,
        gauss,
        imagenet_norm,
    ])
    
    # Pipeline de evaluación (sin augmentaciones)
    eval_pipeline = transforms.Compose([
        to_3ch,
        resize_224,
        to_tensor,
        imagenet_norm,
    ])
    
    return train_pipeline, eval_pipeline


def relabel_dataset(example):
    """Recodifica las etiquetas si es necesario"""
    if example["label"] == 1:
        example["label"] = 0
    return example


def load_and_prepare_data():
    """Carga y prepara el dataset de Alzheimer"""
    print("Cargando dataset...")
    ds = load_dataset(DATASET_NAME)
    
    # Aplicar recodificación de etiquetas
    ds['train'] = ds['train'].map(relabel_dataset)
    ds['test'] = ds['test'].map(relabel_dataset)
    
    # Obtener nombres de las clases
    label_names = ds['train'].features['label'].names
    print(f"Clases encontradas: {label_names}")
    
    return ds, label_names


def create_metadata_dataframe(ds, split_name, label_names):
    """Crea un DataFrame con metadatos de las imágenes"""
    print("Creando DataFrame de metadatos...")
    rows = []
    
    for i in tqdm(range(len(ds[split_name])), desc="Leyendo metadatos"):
        sample = ds[split_name][i]
        img = sample["image"]
        w, h = img.size
        lbl_id = sample["label"]
        lbl_name = label_names[lbl_id]
        fpath = getattr(img, "filename", "") or ""
        fname = os.path.basename(fpath) if fpath else f"idx_{i}.png"
        
        rows.append({
            'index': i,
            'label_id': lbl_id,
            'label': lbl_name,
            'width': w,
            'height': h,
            'filename': fname,
            'filepath': fpath
        })
    
    return pd.DataFrame(rows)


def analyze_data_quality(ds, split_name):
    """Analiza la calidad de las imágenes"""
    print("Analizando calidad de datos...")
    
    def to_gray_float01(pil_img):
        g = pil_img.convert("L")
        arr = np.asarray(g, dtype=np.float32)
        arr = arr / 255.0
        return arr
    
    corrupt_idx = []
    low_var_idx = []
    stats = []
    
    for i in tqdm(range(len(ds[split_name])), desc="Análisis de calidad"):
        pil = ds[split_name][i]["image"]
        
        # Verificar corrupción
        ok = True
        try:
            if getattr(pil, "filename", ""):
                Image.open(pil.filename).verify()
        except Exception:
            ok = False
        
        if not ok:
            corrupt_idx.append(i)
            continue
        
        # Calcular estadísticas
        arr = to_gray_float01(pil)
        mu = float(arr.mean())
        sd = float(arr.std())
        var = float(sd * sd)
        pct_black = float((arr < 0.02).mean())
        pct_white = float((arr > 0.98).mean())
        
        if sd < 0.02:
            low_var_idx.append(i)
        
        stats.append((i, mu, sd, var, pct_black, pct_white))
    
    stats_df = pd.DataFrame(stats, columns=["idx", "mean", "std", "var", "pct_black", "pct_white"])
    
    print(f"Imágenes corruptas: {len(corrupt_idx)}")
    print(f"Imágenes de baja varianza: {len(low_var_idx)}")
    
    return stats_df, corrupt_idx, low_var_idx


def create_data_splits(ds, label_names):
    """Crea los splits de entrenamiento, validación y test"""
    print("Creando splits de datos...")
    
    hf_train = ds["train"]
    hf_test = ds["test"]
    
    # Etiquetas para estratificación
    y_train_all = np.array([hf_train[i]["label"] for i in range(len(hf_train))])
    
    # Split estratificado train/val
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VAL_SPLIT_SIZE, random_state=SEED)
    train_idx, val_idx = next(sss.split(np.arange(len(hf_train)), y_train_all))
    
    # Convertir a int
    train_idx = [int(i) for i in train_idx.tolist()]
    val_idx = [int(i) for i in val_idx.tolist()]
    test_idx = [int(i) for i in range(len(hf_test))]
    
    print(f"Tamaños → Train={len(train_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")
    
    # Mostrar distribución por clase
    def show_class_distribution(hf_split, idx, split_name):
        labs = pd.Series([hf_split[int(i)]["label"] for i in idx])
        vc = labs.value_counts().sort_index()
        vc.index = [label_names[i] for i in vc.index]
        print(f"\nDistribución {split_name}:")
        print(vc)
        return vc
    
    train_dist = show_class_distribution(hf_train, train_idx, "Train")
    val_dist = show_class_distribution(hf_train, val_idx, "Val")
    test_dist = show_class_distribution(hf_test, test_idx, "Test")
    
    return (hf_train, hf_test), (train_idx, val_idx, test_idx), (train_dist, val_dist, test_dist)


def create_data_loaders(hf_splits, indices_splits, transforms_tuple):
    """Crea los DataLoaders para entrenamiento"""
    hf_train, hf_test = hf_splits
    train_idx, val_idx, test_idx = indices_splits
    train_pipeline, eval_pipeline = transforms_tuple
    
    # Crear datasets
    train_ds = HFDataset(hf_train, train_idx, transform=train_pipeline)
    val_ds = HFDataset(hf_train, val_idx, transform=eval_pipeline)
    test_ds = HFDataset(hf_test, test_idx, transform=eval_pipeline)
    
    # Crear DataLoaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    test_loader = DataLoader(
        test_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=PIN_MEMORY
    )
    
    print(f"DataLoaders creados - GPU: {torch.cuda.is_available()}")
    print(f"Batch size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
    
    return train_loader, val_loader, test_loader


def visualize_augmentations(ds, label_names, num_examples=3):
    """Visualiza las augmentaciones aplicadas"""
    train_pipeline, eval_pipeline = get_transforms()
    
    def denorm_imgnet(t):
        mean = torch.tensor(IMAGENET_MEAN, dtype=t.dtype, device=t.device).view(3,1,1)
        std = torch.tensor(IMAGENET_STD, dtype=t.dtype, device=t.device).view(3,1,1)
        x = t * std + mean
        return torch.clamp(x, 0, 1)
    
    split_name = list(ds.keys())[0]
    
    for example_idx in range(num_examples):
        sample = ds[split_name][example_idx]
        pil = sample['image']
        label_id = int(sample['label'])
        label_name = label_names[label_id]
        
        imgs = []
        titles = []
        
        # Original
        orig_vis = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])(pil)
        imgs.append(orig_vis)
        titles.append("Original")
        
        # Augmentaciones
        for k in range(5):
            x = train_pipeline(pil)
            x_vis = denorm_imgnet(x)
            imgs.append(x_vis)
            titles.append(f"Aug {k+1}")
        
        # Sin augmentaciones
        x_eval = eval_pipeline(pil)
        x_eval_vis = denorm_imgnet(x_eval)
        imgs.append(x_eval_vis)
        titles.append("Eval")
        
        # Mostrar
        n = len(imgs)
        cols = 4
        rows = int(np.ceil(n/cols))
        plt.figure(figsize=(4*cols, 4*rows))
        
        for i, im in enumerate(imgs):
            plt.subplot(rows, cols, i+1)
            plt.imshow(np.moveaxis(im.numpy(), 0, 2))
            plt.title(titles[i], fontsize=10)
            plt.axis('off')
        
        plt.suptitle(f"Ejemplo {example_idx+1} - Clase: {label_name}", y=1.02, fontsize=12)
        plt.tight_layout()
        plt.show()