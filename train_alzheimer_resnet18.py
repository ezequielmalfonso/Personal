#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Entrenamiento de ResNet18 para clasificación de Alzheimer en MRI 2D
- Dataset: Falah/Alzheimer_MRI (HuggingFace Datasets)
- Preprocesado: grises→3ch, resize 224, normalización ImageNet, aumentos suaves en train
- Métricas: loss/acc/F1/ROC-AUC, reporte sklearn, matrices de confusión
- Guardados: best_model.pt, métricas por época (CSV), predicciones de test (CSV), gráficos PNG, args y metadatos JSON

Ejecución mínima:
  python train_alzheimer_resnet18.py

Con parámetros:
  python train_alzheimer_resnet18.py --epochs 10 --batch-size 64 --lr 3e-4 --noise-std 0.02

"""

import argparse
import json
import os
import random
import time
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


# ------------------------ Utilidades ------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ------------------------ Transforms ------------------------

class AddGaussianNoise:
    def __init__(self, std: float = 0.02) -> None:
        self.std = float(std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return x
        return torch.clamp(x + torch.randn_like(x) * self.std, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std})"


def build_transforms(noise_std: float, use_bilinear: bool) -> Tuple[transforms.Compose, transforms.Compose]:
    to_3ch = transforms.Grayscale(num_output_channels=3)
    resize_224 = transforms.Resize((224, 224), InterpolationMode.BILINEAR if use_bilinear else InterpolationMode.NEAREST)

    rot_peq = transforms.RandomRotation(
        degrees=10,
        interpolation=InterpolationMode.BILINEAR if use_bilinear else InterpolationMode.NEAREST,
    )
    shift_peq = transforms.RandomAffine(
        degrees=0, translate=(0.05, 0.05),
        interpolation=InterpolationMode.BILINEAR if use_bilinear else InterpolationMode.NEAREST,
    )
    jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)

    to_tensor_01 = transforms.ToTensor()
    gauss = AddGaussianNoise(std=noise_std)
    imagenet_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_pipeline = transforms.Compose([
        to_3ch,
        resize_224,
        rot_peq,
        shift_peq,
        jitter,
        to_tensor_01,
        gauss,
        imagenet_norm,
    ])
    eval_pipeline = transforms.Compose([
        to_3ch,
        resize_224,
        to_tensor_01,
        imagenet_norm,
    ])
    return train_pipeline, eval_pipeline


# ------------------------ Dataset ------------------------

class HFDataset(Dataset):
    def __init__(self, hf_split, indices: List[int], transform: transforms.Compose) -> None:
        self.hf = hf_split
        self.indices = [int(i) for i in indices]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        i = self.indices[idx]
        s = self.hf[int(i)]
        x = self.transform(s["image"])  # PIL -> tensor
        y = int(s["label"])
        return x, y, i


def maybe_collapse_labels(ds, collapse_very_mild_into_mild: bool):
    if not collapse_very_mild_into_mild:
        return ds

    # Intención: fusionar "Very Mild Demented" en "Mild Demented" si existiera ese esquema
    def relabel(example):
        # Si la etiqueta es 1 (Very Mild), mapearla a 0 (Mild). Ajustar según dataset real.
        if example.get("label", None) == 1:
            example["label"] = 0
        return example

    for split in list(ds.keys()):
        ds[split] = ds[split].map(relabel)
    return ds


def build_splits(ds, val_size: float, seed: int) -> Tuple:
    hf_train = ds["train"]
    hf_test = ds.get("test", None)
    if hf_test is None:
        raise RuntimeError("El dataset no tiene split 'test'.")

    label_names = hf_train.features["label"].names
    y_train_all = np.array([int(hf_train[i]["label"]) for i in range(len(hf_train))])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(sss.split(np.arange(len(hf_train)), y_train_all))

    train_idx = [int(i) for i in train_idx.tolist()]
    val_idx = [int(i) for i in val_idx.tolist()]
    test_idx = [int(i) for i in range(len(hf_test))]

    return hf_train, hf_test, label_names, train_idx, val_idx, test_idx


def counts_by_class(hf_split, idx: List[int], label_names: List[str]) -> pd.Series:
    labs = pd.Series([int(hf_split[int(i)]["label"]) for i in idx])
    vc = labs.value_counts().sort_index()
    vc.index = [label_names[i] for i in vc.index]
    return vc


# ------------------------ Modelo ------------------------

def build_model(num_classes: int, freeze_backbone: bool) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False

    in_f = model.fc.in_features
    model.fc = nn.Linear(in_f, num_classes)
    return model


# ------------------------ Entrenamiento ------------------------

def compute_class_weights(hf_train, train_idx: List[int], num_classes: int) -> torch.Tensor:
    y_train = pd.Series([int(hf_train[i]["label"]) for i in train_idx])
    counts = y_train.value_counts().sort_index()
    # Si alguna clase no aparece, evitar división por cero asignando peso 0
    freq = torch.tensor([counts.get(i, 0) for i in range(num_classes)], dtype=torch.float32)
    inv = torch.where(freq > 0, 1.0 / freq, torch.zeros_like(freq))
    # Normalizar para que el promedio sea ~1
    if inv.sum() > 0:
        inv = inv / inv.sum() * num_classes
    return inv


def train_one_epoch(model, loader, optimizer, criterion, device, use_amp: bool) -> float:
    model.train()
    total_loss, n = 0.0, 0
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for xb, yb, _ in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, label_names: List[str]) -> Dict[str, object]:
    model.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred, y_prob = [], [], []

    for xb, yb, _ in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        probs = torch.softmax(logits, dim=1)
        y_true.append(yb.cpu().numpy())
        y_pred.append(probs.argmax(dim=1).cpu().numpy())
        y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true) if y_true else np.array([])
    y_pred = np.concatenate(y_pred) if y_pred else np.array([])
    y_prob = np.concatenate(y_prob) if y_prob else np.empty((0, len(label_names)))

    if y_true.size == 0:
        return {"loss": total_loss / max(n, 1), "acc": np.nan, "f1_macro": np.nan, "auc_macro": np.nan,
                "report_txt": "", "cm": np.zeros((len(label_names), len(label_names)), dtype=int)}

    acc = float((y_true == y_pred).mean())
    f1m = float(f1_score(y_true, y_pred, average="macro"))
    try:
        aucm = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        aucm = float("nan")
    rep_txt = classification_report(y_true, y_pred, target_names=label_names, digits=3)
    cm = confusion_matrix(y_true, y_pred)
    return {"loss": total_loss / max(n, 1), "acc": acc, "f1_macro": f1m, "auc_macro": aucm,
            "report_txt": rep_txt, "cm": cm, "y_true": y_true, "y_pred": y_pred, "y_prob": y_prob}


# ------------------------ Gráficos y guardados ------------------------

def plot_confusion(cm: np.ndarray, labels: List[str], normalize: bool, title: str, out_path: str) -> None:
    M = cm.astype(float)
    if normalize:
        row_sums = M.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        M = M / row_sums
        fmt = ".2f"
        cmap = "Blues"
    else:
        fmt = ".0f"
        cmap = "Oranges"
    plt.figure(figsize=(6, 5))
    sns.heatmap(M, annot=True, fmt=fmt, cmap=cmap, xticklabels=labels, yticklabels=labels, cbar=True)
    plt.title(title)
    plt.ylabel("Verdad")
    plt.xlabel("Predicción")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_training_curves(history: List[Dict[str, float]], out_path: str) -> None:
    if not history:
        return
    epochs = np.arange(1, len(history) + 1)
    tr_loss = [h["train_loss"] for h in history]
    va_loss = [h["val_loss"] for h in history]
    va_acc  = [h["val_acc"] for h in history]
    va_f1   = [h["val_f1"] for h in history]

    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].plot(epochs, tr_loss, label="train_loss")
    axs[0].plot(epochs, va_loss, label="val_loss")
    axs[0].set_title("Pérdida")
    axs[0].set_xlabel("Época")
    axs[0].legend()

    axs[1].plot(epochs, va_acc, label="val_acc")
    axs[1].plot(epochs, va_f1, label="val_f1")
    axs[1].set_title("Validación")
    axs[1].set_xlabel("Época")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, labels: List[str], out_path: str) -> None:
    num_classes = len(labels)
    plt.figure(figsize=(6, 5))
    plotted = False
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(int)
        if y_true_c.sum() == 0 or y_true_c.sum() == len(y_true_c):
            continue
        try:
            fpr, tpr, _ = roc_curve(y_true_c, y_prob[:, c])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{labels[c]} (AUC={roc_auc:.2f})")
            plotted = True
        except Exception:
            continue
    plt.plot([0, 1], [0, 1], "k--", label="Azar")
    plt.xlabel("1 - Especificidad (FPR)")
    plt.ylabel("Sensibilidad (TPR)")
    plt.title("Curvas ROC por clase (test)")
    if plotted:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_augmentations_grid(sample_pil, train_pipeline: transforms.Compose, eval_pipeline: transforms.Compose, label_name: str, out_path: str) -> None:
    def denorm_imgnet(t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=t.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=t.dtype).view(3, 1, 1)
        x = t * std + mean
        return torch.clamp(x, 0, 1)

    imgs, titles = [], []
    orig_vis = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224), InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])(sample_pil)
    imgs.append(orig_vis)
    titles.append("Original (224)")

    for k in range(5):
        x = train_pipeline(sample_pil)
        imgs.append(denorm_imgnet(x))
        titles.append(f"Augment {k+1}")

    x_eval = eval_pipeline(sample_pil)
    imgs.append(denorm_imgnet(x_eval))
    titles.append("Eval (sin aumentos)")

    n = len(imgs)
    cols = 4
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, im in enumerate(imgs):
        plt.subplot(rows, cols, i + 1)
        arr = np.moveaxis(im.numpy(), 0, 2)
        plt.imshow(arr, cmap="gray")
        plt.title(titles[i], fontsize=10)
        plt.axis("off")
    plt.suptitle(f"Clase: {label_name}", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ------------------------ Inferencia ------------------------

def run_single_image_inference(
    model: nn.Module,
    image_path: str,
    eval_tf: transforms.Compose,
    device: torch.device,
    label_names: List[str],
    out_dir: str,
) -> None:
    pil = Image.open(image_path)
    x = eval_tf(pil).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    pred_name = label_names[pred_idx]

    # Guardar JSON
    info = {
        "image_path": image_path,
        "pred_idx": pred_idx,
        "pred_label": pred_name,
        "probs": {label_names[i]: float(probs[i]) for i in range(len(label_names))},
    }
    with open(os.path.join(out_dir, "inference.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    # Guardar figura
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(pil.resize((224, 224)))
    axs[0].set_title(f"Pred: {pred_name}")
    axs[0].axis("off")
    axs[1].bar(range(len(label_names)), probs, tick_label=label_names)
    axs[1].set_ylim(0, 1)
    axs[1].set_title("Probabilidades")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "inference.png"))
    plt.close(fig)
    print(f"Inferencia guardada en inference.png e inference.json | Pred: {pred_name}")


def run_dataset_index_inference(
    model: nn.Module,
    ds_split,
    index: int,
    eval_tf: transforms.Compose,
    device: torch.device,
    label_names: List[str],
    out_dir: str,
) -> None:
    sample = ds_split[int(index)]
    pil = sample["image"]
    true_idx = int(sample["label"])
    true_name = label_names[true_idx]
    x = eval_tf(pil).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    pred_name = label_names[pred_idx]

    info = {
        "dataset_index": int(index),
        "true_idx": true_idx,
        "true_label": true_name,
        "pred_idx": pred_idx,
        "pred_label": pred_name,
        "probs": {label_names[i]: float(probs[i]) for i in range(len(label_names))},
    }
    with open(os.path.join(out_dir, "inference_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(pil.resize((224, 224)))
    axs[0].set_title(f"True: {true_name}\nPred: {pred_name}")
    axs[0].axis("off")
    axs[1].bar(range(len(label_names)), probs, tick_label=label_names)
    axs[1].set_ylim(0, 1)
    axs[1].set_title("Probabilidades")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "inference_dataset.png"))
    plt.close(fig)
    print(f"Inferencia de dataset guardada en inference_dataset.png e inference_dataset.json | Pred: {pred_name} (True: {true_name})")

# ------------------------ Main ------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena ResNet18 en Falah/Alzheimer_MRI")
    p.add_argument("--dataset", type=str, default="Falah/Alzheimer_MRI", help="Nombre del dataset en HF")
    p.add_argument("--output-dir", type=str, default="outputs", help="Directorio base de salidas")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=0, help="0 = auto (64 si GPU, 16 si CPU)")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--noise-std", type=float, default=0.02, help="Ruido gaussiano en [0,1] (train)")
    p.add_argument("--no-noise", action="store_true", help="Desactiva ruido gaussiano")
    p.add_argument("--nearest", action="store_true", help="Usar Interpolation NEAREST en lugar de bilinear")
    p.add_argument("--freeze-backbone", action="store_true", help="Congelar feature extractor y entrenar solo la FC")
    p.add_argument("--no-amp", action="store_true", help="Desactiva AMP aún con GPU disponible")
    p.add_argument("--max-train-steps", type=int, default=0, help="Límite de steps por época (0 = sin límite)")
    p.add_argument("--collapse-very-mild", action="store_true", help="Fusiona Very Mild en Mild si aplica")
    p.add_argument("--infer-image", type=str, default="", help="Ruta a imagen PNG/JPG para inferencia")
    p.add_argument("--infer-index", type=int, default=-1, help="Índice del split test para inferencia/visualización")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    device = get_device()

    if args.no_noise:
        args.noise_std = 0.0
    use_bilinear = not args.nearest
    use_amp = (device.type == "cuda") and (not args.no_amp)

    batch_size = args.batch_size
    if batch_size <= 0:
        batch_size = 64 if device.type == "cuda" else 16

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, run_id)
    ensure_dir(run_dir)

    # Guardar args y metadatos
    meta = {
        "args": vars(args),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Dispositivo: {device} | AMP: {use_amp} | batch_size={batch_size}")

    # Dataset
    print("Cargando dataset...", flush=True)
    ds = load_dataset(args.dataset)
    ds = maybe_collapse_labels(ds, args.collapse_very_mild)

    hf_train, hf_test, label_names, train_idx, val_idx, test_idx = build_splits(ds, args.val_size, args.seed)

    print(f"Clases: {label_names} (num_classes={len(label_names)})")
    print(f"Tamaños → Train={len(train_idx)}  Val={len(val_idx)}  Test={len(test_idx)}")
    print("Conteos por clase (Train):\n", counts_by_class(hf_train, train_idx, label_names))
    print("Conteos por clase (Val):\n", counts_by_class(hf_train, val_idx, label_names))
    print("Conteos por clase (Test):\n", counts_by_class(hf_test, test_idx, label_names))

    # Transforms
    train_tf, eval_tf = build_transforms(noise_std=args.noise_std, use_bilinear=use_bilinear)

    # Datasets y loaders
    train_ds = HFDataset(hf_train, train_idx, transform=train_tf)
    val_ds = HFDataset(hf_train, val_idx, transform=eval_tf)
    test_ds = HFDataset(hf_test, test_idx, transform=eval_tf)

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_mem)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_mem)

    # Modelo
    num_classes = len(label_names)
    model = build_model(num_classes=num_classes, freeze_backbone=args.freeze_backbone).to(device)

    # Pesos por clase
    class_weights = compute_class_weights(hf_train, train_idx, num_classes)
    print("Pesos por clase:", dict(zip(label_names, class_weights.tolist())))
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)
    except TypeError:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # Warm-up DataLoader
    t0 = time.time()
    _wu = next(iter(train_loader))
    t1 = time.time()
    print(f"Warm-up: primer batch en {t1 - t0:.2f}s | shape={tuple(_wu[0].shape)}")

    # Entrenamiento
    history: List[Dict[str, float]] = []
    best_f1, best_epoch = -1.0, -1
    max_steps = args.max_train_steps if args.max_train_steps and args.max_train_steps > 0 else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, seen, steps = 0.0, 0, 0
        for step, (xb, yb, _) in enumerate(train_loader):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            if use_amp:
                scaler = torch.cuda.amp.GradScaler()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            seen += xb.size(0)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        tr_loss = total_loss / max(seen, 1)

        val_stats = evaluate(model, val_loader, criterion, device, label_names)
        try:
            scheduler.step(val_stats["f1_macro"])  # observar F1 macro
        except Exception:
            pass

        lr = optimizer.param_groups[0]['lr']
        print(
            f"[Epoch {epoch:02d}] train_loss={tr_loss:.4f} | "
            f"val_loss={val_stats['loss']:.4f} | acc={val_stats['acc']:.3f} | "
            f"f1_macro={val_stats['f1_macro']:.3f} | auc_macro={val_stats['auc_macro']:.3f} | lr={lr:.1e}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": float(val_stats["loss"]),
            "val_acc": float(val_stats["acc"]),
            "val_f1": float(val_stats["f1_macro"]),
            "val_auc": float(val_stats["auc_macro"]),
            "lr": float(lr),
        })

        if val_stats["f1_macro"] > best_f1:
            best_f1, best_epoch = float(val_stats["f1_macro"]), epoch
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

    print(f"Mejor F1_macro en epoch {best_epoch}: {best_f1:.3f} → guardado best_model.pt")

    # Guardar history CSV y curvas
    hist_df = pd.DataFrame(history)
    hist_csv = os.path.join(run_dir, "history.csv")
    hist_df.to_csv(hist_csv, index=False)
    plot_training_curves(history, os.path.join(run_dir, "training_curves.png"))

    # Evaluación en test
    if os.path.exists(os.path.join(run_dir, "best_model.pt")):
        model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pt"), map_location=device))
    model.eval()

    test_stats = evaluate(model, test_loader, criterion, device, label_names)

    # Guardar reporte y matrices de confusión
    with open(os.path.join(run_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(test_stats["report_txt"])

    cm = test_stats["cm"]
    plot_confusion(cm, label_names, normalize=False, title="Matriz de confusión (abs)", out_path=os.path.join(run_dir, "confusion_abs.png"))
    plot_confusion(cm, label_names, normalize=True, title="Matriz de confusión (norm)", out_path=os.path.join(run_dir, "confusion_norm.png"))

    # ROC y AUC
    try:
        auc_macro = roc_auc_score(test_stats["y_true"], test_stats["y_prob"], multi_class="ovr", average="macro")
        print(f"AUC macro (test): {auc_macro:.3f}")
    except Exception as e:
        print("AUC macro no disponible:", e)
        auc_macro = float("nan")
    plot_roc_curves(test_stats["y_true"], test_stats["y_prob"], label_names, out_path=os.path.join(run_dir, "roc_curves.png"))

    # Guardar predicciones
    preds_df = pd.DataFrame({
        "true": [label_names[int(i)] for i in test_stats["y_true"]],
        "pred": [label_names[int(i)] for i in test_stats["y_pred"]],
    })
    # Probabilidades por clase
    for c, name in enumerate(label_names):
        preds_df[f"prob_{name}"] = test_stats["y_prob"][:, c]
    preds_df.to_csv(os.path.join(run_dir, "test_predictions.csv"), index=False)

    # Guardar una grilla de aumentaciones para la presentación
    try:
        sample = ds["train"][0]
        save_augmentations_grid(
            sample_pil=sample["image"],
            train_pipeline=train_tf,
            eval_pipeline=eval_tf,
            label_name=label_names[int(sample["label"])],
            out_path=os.path.join(run_dir, "augmentations_grid.png"),
        )
    except Exception as e:
        print("No se pudo guardar grilla de aumentaciones:", e)

    # Inferencia opcional para la presentación
    try:
        if args.infer_image:
            run_single_image_inference(
                model=model,
                image_path=args.infer_image,
                eval_tf=eval_tf,
                device=device,
                label_names=label_names,
                out_dir=run_dir,
            )
        if args.infer_index is not None and int(args.infer_index) >= 0:
            run_dataset_index_inference(
                model=model,
                ds_split=ds["test"],
                index=int(args.infer_index),
                eval_tf=eval_tf,
                device=device,
                label_names=label_names,
                out_dir=run_dir,
            )
    except Exception as e:
        print("Inferencia opcional falló:", e)

    # Resumen JSON de métricas
    summary = {
        "best_epoch": int(best_epoch),
        "best_val_f1_macro": float(best_f1),
        "test_acc": float(test_stats["acc"]),
        "test_f1_macro": float(test_stats["f1_macro"]),
        "test_auc_macro": float(auc_macro) if not np.isnan(auc_macro) else None,
        "label_names": list(label_names),
        "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nListo. Archivos clave para la presentación:")
    print("- best_model.pt")
    print("- training_curves.png")
    print("- confusion_abs.png y confusion_norm.png")
    print("- roc_curves.png (si aplica)")
    print("- classification_report.txt y test_predictions.csv")
    print("- summary.json e history.csv")
    print(f"Directorio del run: {run_dir}")


if __name__ == "__main__":
    main()
