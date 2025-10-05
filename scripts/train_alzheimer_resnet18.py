#!/usr/bin/env python3
"""
Entrenamiento y evaluación de ResNet18 para clasificación de Alzheimer
usando el dataset Hugging Face "Falah/Alzheimer_MRI".

Características clave:
- Carga del dataset HF (splits train/test) y creación de validación estratificada
- Preprocesamiento (grises→3ch, resize 224, normalización ImageNet) y aumentos suaves en train
- ResNet18 preentrenada en ImageNet, con reentrenamiento de la capa final
- Ponderación de clases por frecuencia (para mitigar desbalance)
- Métricas en val/test: loss, acc, F1 macro, AUC macro, reporte y matrices de confusión
- Guardado de mejor checkpoint por F1 macro de validación
- Figuras de presentación: muestras de aumentos, matrices de confusión y ROC por clase

Uso rápido:
  python scripts/train_alzheimer_resnet18.py --epochs 10 --output_dir outputs

Requiere dependencias listadas en requirements.txt
"""

import os
import json
import random
import argparse
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms import InterpolationMode
from tqdm import tqdm
from PIL import Image

from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc as sk_auc,
)


# ------------------------- Utilidades generales -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class AddGaussianNoise:
    def __init__(self, std: float = 0.02):
        self.std = float(std)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return x
        return torch.clamp(x + torch.randn_like(x) * self.std, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std})"


# ------------------------- Dataset y Dataloaders -------------------------

class HFDataset(Dataset):
    def __init__(self, hf_split, indices: List[int], transform):
        self.hf = hf_split
        self.idx = [int(i) for i in indices]
        self.tfm = transform

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, k: int):
        i = self.idx[k]
        s = self.hf[i]
        x = self.tfm(s["image"])  # PIL → Tensor
        y = int(s["label"])       # entero 0..C-1
        return x, y, i


def build_transforms(img_size: int = 224, use_gaussian_noise: bool = True, noise_std: float = 0.02) -> Tuple[transforms.Compose, transforms.Compose]:
    to_3ch = transforms.Grayscale(num_output_channels=3)
    resize_224 = transforms.Resize((img_size, img_size), InterpolationMode.BILINEAR)
    rot_peq = transforms.RandomRotation(
        degrees=10, interpolation=InterpolationMode.BILINEAR
    )
    shift_peq = transforms.RandomAffine(
        degrees=0, translate=(0.05, 0.05), interpolation=InterpolationMode.BILINEAR
    )
    jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1)
    to_tensor_01 = transforms.ToTensor()
    gauss = AddGaussianNoise(std=noise_std) if use_gaussian_noise else transforms.Lambda(lambda t: t)
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


def stratified_train_val_indices(hf_train, val_size: float, seed: int) -> Tuple[List[int], List[int]]:
    y_train_all = np.array([int(hf_train[i]["label"]) for i in range(len(hf_train))])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(sss.split(np.arange(len(hf_train)), y_train_all))
    return train_idx.tolist(), val_idx.tolist()


def build_dataloaders(hf_train, hf_test, train_idx: List[int], val_idx: List[int],
                      train_tfm, eval_tfm, batch_size: int, num_workers: int,
                      pin_memory: bool) -> Tuple[DataLoader, DataLoader, DataLoader]:
    test_idx = list(range(len(hf_test)))

    train_ds = HFDataset(hf_train, train_idx, transform=train_tfm)
    val_ds   = HFDataset(hf_train, val_idx,   transform=eval_tfm)
    test_ds  = HFDataset(hf_test,  test_idx,  transform=eval_tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


# ------------------------- Modelo y entrenamiento -------------------------

def build_resnet18(num_classes: int, device: torch.device, freeze_backbone: bool = False) -> nn.Module:
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
    return model.to(device)


def class_weights_from_indices(hf_split, indices: List[int], num_classes: int) -> torch.Tensor:
    labels = [int(hf_split[i]["label"]) for i in indices]
    counts = pd.Series(labels).value_counts().sort_index()
    # Si falta alguna clase en train, completar con 0
    for c in range(num_classes):
        if c not in counts.index:
            counts.loc[c] = 0
    counts = counts.sort_index().values
    counts = np.where(counts == 0, 1, counts)  # evitar división por cero
    weights = 1.0 / torch.tensor(counts, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes
    return weights


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer, criterion, device: torch.device,
                    use_amp: bool = True, max_steps: Optional[int] = None) -> float:
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    running_loss, seen = 0.0, 0

    total_steps = len(loader) if max_steps is None else min(len(loader), max_steps)
    pbar = tqdm(enumerate(loader), total=total_steps, desc="Train", leave=False)
    for step, (xb, yb, _) in pbar:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * xb.size(0)
        seen += xb.size(0)
        pbar.set_postfix(loss=running_loss / max(seen, 1))

        if max_steps is not None and (step + 1) >= max_steps:
            break

    return running_loss / max(seen, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device, label_names: List[str]):
    model.eval()
    total_loss, n = 0.0, 0
    y_true, y_pred, y_prob = [], [], []

    for xb, yb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)

        probs = torch.softmax(logits, dim=1)
        y_true.append(yb.cpu().numpy())
        y_pred.append(probs.argmax(dim=1).cpu().numpy())
        y_prob.append(probs.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    acc = float((y_true == y_pred).mean())
    f1m = float(f1_score(y_true, y_pred, average="macro"))
    try:
        aucm = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        aucm = float("nan")

    rep_txt = classification_report(y_true, y_pred, target_names=label_names, digits=3)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "loss": total_loss / max(n, 1),
        "acc": acc,
        "f1_macro": f1m,
        "auc_macro": aucm,
        "report_txt": rep_txt,
        "cm": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }


# ------------------------- Visualización y guardado -------------------------

def save_confusion_matrices(cm: np.ndarray, labels: List[str], out_dir: str, prefix: str = "confusion") -> None:
    def _plot(M: np.ndarray, normalize: bool, title: str, fname: str) -> None:
        if normalize:
            M = M.astype("float")
            M = M / M.sum(axis=1, keepdims=True).clip(min=1)
            fmt, cmap = ".2f", "Blues"
        else:
            fmt, cmap = "d", "Oranges"

        plt.figure(figsize=(6, 5))
        plt.imshow(M, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        ticks = np.arange(len(labels))
        plt.xticks(ticks, labels, rotation=45, ha="right")
        plt.yticks(ticks, labels)
        thresh = M.max() / 2 if M.size else 0.5
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                plt.text(j, i, format(M[i, j], fmt), ha="center", va="center",
                         color="white" if M[i, j] > thresh else "black")
        plt.ylabel("Verdad")
        plt.xlabel("Predicción")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close()

    _plot(cm, False, "Matriz de confusión (abs)", f"{prefix}_abs.png")
    _plot(cm, True,  "Matriz de confusión (norm)", f"{prefix}_norm.png")


def save_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, labels: List[str], out_path: str) -> None:
    num_classes = len(labels)
    plt.figure(figsize=(6, 5))
    plotted_any = False
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(int)
        if y_true_c.sum() == 0 or y_true_c.sum() == len(y_true_c):
            continue
        fpr, tpr, _ = roc_curve(y_true_c, y_prob[:, c])
        roc_auc = sk_auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{labels[c]} (AUC={roc_auc:.2f})")
        plotted_any = True
    plt.plot([0, 1], [0, 1], "k--", label="Azar")
    plt.xlabel("1 - Especificidad (FPR)")
    plt.ylabel("Sensibilidad (TPR)")
    plt.title("Curvas ROC por clase (test)")
    if plotted_any:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_sample_augmentations(hf_split, label_names: List[str], train_tfm, eval_tfm, out_path: str, img_size: int = 224) -> None:
    # tomar una imagen del split (ej: train)
    sample = hf_split[0]
    pil = sample["image"]
    lbl = int(sample["label"]) if "label" in sample else None
    label_title = label_names[lbl] if (lbl is not None and 0 <= lbl < len(label_names)) else ""

    def denorm_imgnet(t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=t.dtype).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=t.dtype).view(3, 1, 1)
        x = t * std + mean
        return torch.clamp(x, 0, 1)

    to_3ch = transforms.Grayscale(num_output_channels=3)
    resize_ = transforms.Resize((img_size, img_size), InterpolationMode.BILINEAR)
    orig_vis = transforms.Compose([to_3ch, resize_, transforms.ToTensor()])(pil)

    imgs = [orig_vis]
    titles = ["Original (224)"]
    for k in range(5):
        x = train_tfm(pil)
        imgs.append(denorm_imgnet(x))
        titles.append(f"Augment {k+1}")

    x_eval = eval_tfm(pil)
    imgs.append(denorm_imgnet(x_eval))
    titles.append("Eval (sin aumentos)")

    n = len(imgs)
    cols = 4
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(4 * cols, 4 * rows))
    for i, im in enumerate(imgs):
        plt.subplot(rows, cols, i + 1)
        np_img = np.moveaxis(im.numpy(), 0, 2)
        plt.imshow(np_img)
        plt.title(titles[i], fontsize=10)
        plt.axis("off")
    if label_title:
        plt.suptitle(f"Clase: {label_title}", y=1.02, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_class_distribution(hf_train, hf_test, tr_idx: List[int], val_idx: List[int], labels: List[str], out_path: str) -> None:
    """Guarda figura con distribución de clases en train/val/test.

    Muestra tres subplots con conteos por clase para cada split.
    """
    def split_counts(hf_split, indices) -> pd.Series:
        labs = pd.Series([int(hf_split[int(i)]["label"]) for i in indices])
        vc = labs.value_counts().sort_index()
        # asegurar todas las clases presentes (con 0 si falta alguna)
        for c in range(len(labels)):
            if c not in vc.index:
                vc.loc[c] = 0
        vc = vc.sort_index()
        vc.index = [labels[i] for i in vc.index]
        return vc

    tr_vc = split_counts(hf_train, tr_idx)
    val_vc = split_counts(hf_train, val_idx)
    test_vc = split_counts(hf_test, range(len(hf_test)))

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    tr_vc.plot(kind="bar", ax=axs[0], color="#4e79a7")
    axs[0].set_title("Train")
    axs[0].set_xlabel("Clase")
    axs[0].set_ylabel("Cantidad")

    val_vc.plot(kind="bar", ax=axs[1], color="#59a14f")
    axs[1].set_title("Val")
    axs[1].set_xlabel("Clase")

    test_vc.plot(kind="bar", ax=axs[2], color="#e15759")
    axs[2].set_title("Test")
    axs[2].set_xlabel("Clase")

    plt.suptitle("Distribución de clases por split")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ------------------------- Main -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrenamiento ResNet18 Alzheimer MRI")
    p.add_argument("--output_dir", type=str, default="outputs", help="Directorio de salida")
    p.add_argument("--epochs", type=int, default=5, help="Cantidad de épocas")
    p.add_argument("--batch_size", type=int, default=None, help="Tamaño de batch (auto por dispositivo si None)")
    p.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay AdamW")
    p.add_argument("--val_size", type=float, default=0.15, help="Proporción de validación del split train")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria")
    p.add_argument("--img_size", type=int, default=224, help="Tamaño de imagen (largo=ancho)")
    p.add_argument("--noise_std", type=float, default=0.02, help="Std de ruido gaussiano en [0,1]")
    p.add_argument("--no_noise", action="store_true", help="Desactiva ruido gaussiano")
    p.add_argument("--freeze_backbone", action="store_true", help="Congelar todo menos la FC final")
    p.add_argument("--max_train_steps", type=int, default=None, help="Límite de steps por época (debug)")
    p.add_argument("--no_amp", action="store_true", help="Desactiva AMP incluso en GPU")
    p.add_argument("--relabel_1_to_0", action="store_true", help="Opcional: remap label 1→0")
    return p.parse_args()


def maybe_relabel(split):
    def _fn(example):
        if int(example.get("label", -1)) == 1:
            example["label"] = 0
        return example
    return _fn


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.output_dir)

    has_gpu = torch.cuda.is_available()
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 64 if has_gpu else 16

    num_workers = min(2, os.cpu_count() or 2)
    pin_memory = has_gpu

    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA disponible: {has_gpu}")
    print(f"Batch size: {batch_size} | Workers: {num_workers}")

    # 1) Dataset
    ds = load_dataset("Falah/Alzheimer_MRI")
    if args.relabel_1_to_0:
        for split in ("train", "test"):
            if split in ds:
                ds[split] = ds[split].map(maybe_relabel(ds[split]))

    if "train" not in ds:
        raise RuntimeError("El dataset no tiene split 'train'.")
    if "test" not in ds:
        raise RuntimeError("El dataset no tiene split 'test'.")

    hf_train = ds["train"]
    hf_test = ds["test"]

    if "label" in hf_train.features and hasattr(hf_train.features["label"], "names"):
        label_names = list(hf_train.features["label"].names)
    else:
        # fallback genérico
        num_unique = len(set(int(hf_train[i]["label"]) for i in range(len(hf_train))))
        label_names = [f"class_{i}" for i in range(num_unique)]
    num_classes = len(label_names)

    # 2) Splits
    tr_idx, val_idx = stratified_train_val_indices(hf_train, val_size=args.val_size, seed=args.seed)

    # 3) Transforms
    train_tfm, eval_tfm = build_transforms(
        img_size=args.img_size,
        use_gaussian_noise=(not args.no_noise),
        noise_std=args.noise_std,
    )

    # 4) Dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(
        hf_train, hf_test, tr_idx, val_idx, train_tfm, eval_tfm, batch_size, num_workers, pin_memory
    )

    # 5) Modelo
    model = build_resnet18(num_classes=num_classes, device=device, freeze_backbone=args.freeze_backbone)

    # 6) Loss con pesos por clase (del split train)
    cls_weights = class_weights_from_indices(hf_train, tr_idx, num_classes=num_classes)
    print("Pesos por clase:", {label_names[i]: float(cls_weights[i]) for i in range(num_classes)})
    criterion = nn.CrossEntropyLoss(weight=cls_weights.to(device))

    # 7) Optimizador + scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    try:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)
    except TypeError:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # 8) Warm-up dataloader (detección temprana de cuelgues)
    xb_wu, yb_wu, _ = next(iter(train_loader))
    print(f"Warm-up batch: shape={tuple(xb_wu.shape)} | clases={sorted(set(yb_wu.tolist()))}")

    # 9) Entrenamiento
    best_f1, best_epoch = -1.0, -1
    use_amp = has_gpu and (not args.no_amp)

    history = []
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_amp=use_amp, max_steps=args.max_train_steps
        )
        val_stats = evaluate(model, val_loader, criterion, device, label_names)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"train_loss={tr_loss:.4f} | val_loss={val_stats['loss']:.4f} | "
            f"acc={val_stats['acc']:.3f} | f1_macro={val_stats['f1_macro']:.3f} | "
            f"auc_macro={val_stats['auc_macro']:.3f} | lr={lr:.1e}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "val_f1_macro": val_stats["f1_macro"],
            "val_auc_macro": val_stats["auc_macro"],
        })

        try:
            scheduler.step(val_stats["f1_macro"])
        except Exception:
            pass

        if val_stats["f1_macro"] > best_f1:
            best_f1, best_epoch = val_stats["f1_macro"], epoch
            best_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save({
                "model_state": model.state_dict(),
                "num_classes": num_classes,
                "label_names": label_names,
                "img_size": args.img_size,
            }, best_path)
            print(f"Checkpoint actualizado: {best_path}")

    print(f"\nMejor F1_macro en epoch {best_epoch}: {best_f1:.3f}")

    # 10) Evaluación en test
    # Cargar mejor modelo
    best_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print("Cargado mejor checkpoint para test.")
    model.eval()

    test_stats = evaluate(model, test_loader, criterion, device, label_names)
    print("\n=== TEST — Reporte por clase ===")
    print(test_stats["report_txt"])

    # 11) Guardados de salida para presentación
    # Historial
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Reporte y métricas
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(test_stats["report_txt"]) 
    with open(os.path.join(args.output_dir, "metrics_test.json"), "w") as f:
        json.dump({
            "test_loss": test_stats["loss"],
            "test_acc": test_stats["acc"],
            "test_f1_macro": test_stats["f1_macro"],
            "test_auc_macro": test_stats["auc_macro"],
            "best_val_f1_macro": best_f1,
            "best_epoch": best_epoch,
            "label_names": label_names,
        }, f, indent=2)

    # Matrices de confusión
    save_confusion_matrices(test_stats["cm"], label_names, args.output_dir, prefix="confusion_test")

    # Curvas ROC
    try:
        save_roc_curves(test_stats["y_true"], test_stats["y_prob"], label_names,
                         os.path.join(args.output_dir, "roc_curves_test.png"))
    except Exception as e:
        print("Curvas ROC no disponibles:", e)

    # Muestras de aumentaciones
    try:
        save_sample_augmentations(hf_train, label_names, train_tfm, eval_tfm,
                                  os.path.join(args.output_dir, "sample_augmentations.png"), img_size=args.img_size)
    except Exception as e:
        print("No se pudieron guardar las muestras de aumentaciones:", e)

    # Distribución de clases por split
    try:
        save_class_distribution(
            hf_train, hf_test, tr_idx, val_idx, label_names,
            os.path.join(args.output_dir, "class_distribution.png")
        )
    except Exception as e:
        print("No se pudo guardar la distribución de clases:", e)

    print(f"\nListo. Archivos en: {args.output_dir}")


if __name__ == "__main__":
    main()
