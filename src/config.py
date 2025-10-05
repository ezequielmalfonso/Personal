"""
Configuración del proyecto de detección de Alzheimer
"""
import torch

# Configuración general
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuración del dataset
DATASET_NAME = "Falah/Alzheimer_MRI"
VAL_SPLIT_SIZE = 0.15

# Configuración de entrenamiento
BATCH_SIZE = 64 if torch.cuda.is_available() else 16
NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()

# Configuración del modelo
MODEL_NAME = "resnet18"
NUM_CLASSES = 4  # MildDemented, ModerateDemented, NonDemented, VeryMildDemented
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 15

# Configuración de data augmentation
USE_GAUSSIAN_NOISE = True
GAUSS_STD = 0.02
ROTATION_DEGREES = 10
TRANSLATE_RATIO = 0.05
BRIGHTNESS_FACTOR = 0.1
CONTRAST_FACTOR = 0.1

# Paths
MODEL_SAVE_PATH = "models/best_model.pt"
RESULTS_PATH = "results/"
LOGS_PATH = "logs/"

# ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]