# 🧠 Detección de Alzheimer con ResNet18

Sistema de Machine Learning para detectar niveles de demencia en imágenes de resonancia magnética cerebral utilizando Deep Learning.

## 📋 Descripción del Proyecto

Este proyecto implementa un modelo de clasificación basado en **ResNet18** preentrenado para detectar diferentes niveles de demencia en imágenes MRI del cerebro. El modelo puede clasificar imágenes en 4 categorías:

- **NonDemented**: Sin demencia
- **VeryMildDemented**: Demencia muy leve  
- **MildDemented**: Demencia leve
- **ModerateDemented**: Demencia moderada

## 🎯 Características Principales

- ✅ **Transfer Learning** con ResNet18 preentrenado en ImageNet
- ✅ **Data Augmentation** avanzado para mejorar generalización
- ✅ **Manejo de clases desbalanceadas** con pesos automáticos
- ✅ **Evaluación completa** con múltiples métricas
- ✅ **Sistema de inferencia** fácil de usar
- ✅ **Visualizaciones** para presentaciones
- ✅ **Código modular** y bien documentado

## 🚀 Instalación Rápida

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd alzheimer-detection
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Entrenar el modelo
```bash
python train_alzheimer_model.py
```

### 4. Probar el modelo
```bash
python demo_inference.py
```

## 📁 Estructura del Proyecto

```
alzheimer-detection/
├── src/                          # Código fuente modular
│   ├── config.py                 # Configuración del proyecto
│   ├── data_utils.py             # Utilidades de datos
│   ├── model.py                  # Definición del modelo
│   ├── evaluation.py             # Evaluación y métricas
│   └── inference.py              # Sistema de inferencia
├── models/                       # Modelos entrenados
├── results/                      # Resultados y visualizaciones
├── logs/                         # Logs de entrenamiento
├── train_alzheimer_model.py      # Script principal de entrenamiento
├── demo_inference.py             # Demo de inferencia
├── presentation_demo.py          # Generador de presentación
├── requirements.txt              # Dependencias
└── README.md                     # Este archivo
```

## 🔧 Uso Detallado

### Entrenamiento del Modelo

El script principal entrena el modelo completo:

```bash
python train_alzheimer_model.py
```

**Lo que hace este script:**
1. 📥 Descarga el dataset automáticamente
2. 🔍 Analiza la calidad de los datos
3. ✂️ Crea splits estratificados (train/val/test)
4. 🔄 Aplica data augmentation
5. 🏗️ Crea y configura el modelo ResNet18
6. 🚀 Entrena por múltiples épocas
7. 📊 Evalúa en el conjunto de test
8. 💾 Guarda el mejor modelo
9. 📈 Genera visualizaciones completas

### Inferencia y Predicciones

#### Demo Interactivo
```bash
python demo_inference.py
```

#### Analizar una imagen específica
```bash
python demo_inference.py --image ruta/a/imagen.jpg
```

#### Procesar directorio completo
```bash
python demo_inference.py --demo-dir ruta/a/directorio/
```

### Generar Presentación

Para crear visualizaciones para presentaciones:

```bash
python presentation_demo.py
```

## 📊 Configuración del Modelo

### Parámetros Principales (src/config.py)

```python
# Entrenamiento
EPOCHS = 15                    # Número de épocas
BATCH_SIZE = 64               # Tamaño del lote
LEARNING_RATE = 3e-4          # Tasa de aprendizaje
WEIGHT_DECAY = 1e-4           # Regularización L2

# Data Augmentation
ROTATION_DEGREES = 10         # Rotación aleatoria (±10°)
TRANSLATE_RATIO = 0.05        # Traslación (±5%)
BRIGHTNESS_FACTOR = 0.1       # Brillo (±10%)
CONTRAST_FACTOR = 0.1         # Contraste (±10%)
GAUSS_STD = 0.02             # Ruido gaussiano
```

### Modificar Configuración

Para cambiar parámetros, edita `src/config.py`:

```python
# Ejemplo: Entrenar por más épocas
EPOCHS = 30

# Ejemplo: Usar batch size más pequeño
BATCH_SIZE = 32

# Ejemplo: Cambiar learning rate
LEARNING_RATE = 1e-4
```

## 📈 Métricas y Evaluación

El modelo se evalúa usando múltiples métricas:

- **Accuracy**: Precisión general
- **F1-Macro**: F1 promedio por clase
- **AUC-Macro**: Área bajo curva ROC promedio
- **Matriz de Confusión**: Análisis detallado por clase
- **Curvas ROC**: Rendimiento por clase

### Resultados Típicos

Con la configuración por defecto, puedes esperar:
- **Test Accuracy**: ~85-90%
- **F1-Macro**: ~0.85-0.90
- **AUC-Macro**: ~0.90-0.95

## 🔬 Detalles Técnicos

### Arquitectura del Modelo

- **Base**: ResNet18 preentrenado en ImageNet
- **Entrada**: Imágenes 224×224×3 (RGB)
- **Salida**: 4 clases (probabilidades)
- **Parámetros**: ~11M entrenables

### Preprocesamiento

1. **Conversión**: Escala de grises → RGB (3 canales)
2. **Redimensionamiento**: 224×224 píxeles
3. **Normalización**: Estadísticas de ImageNet
4. **Augmentation**: Rotación, traslación, ruido, ajustes de color

### Data Augmentation

```python
# Transformaciones aplicadas en entrenamiento
- Rotación aleatoria: ±10°
- Traslación: ±5% en X,Y
- Ajuste de brillo: ±10%
- Ajuste de contraste: ±10%
- Ruido gaussiano: σ=0.02
- Normalización ImageNet
```

## 🎨 Uso Programático

### Cargar modelo entrenado

```python
from src.inference import AlzheimerPredictor

# Inicializar predictor
predictor = AlzheimerPredictor("models/best_model.pt")

# Predecir imagen
result = predictor.predict_single_image("imagen.jpg")
print(f"Predicción: {result['predicted_class']}")
print(f"Confianza: {result['confidence']:.1%}")
```

### Entrenar modelo personalizado

```python
from src.model import create_model, train_model
from src.data_utils import create_data_loaders

# Crear modelo
model = create_model(num_classes=4)

# Entrenar (requiere DataLoaders configurados)
history, best_f1, best_epoch = train_model(
    model, train_loader, val_loader, 
    criterion, optimizer, scheduler, 
    label_names, epochs=20
)
```

## 🐛 Solución de Problemas

### Error: "CUDA out of memory"
```python
# En src/config.py, reducir batch size
BATCH_SIZE = 16  # o incluso 8
```

### Error: "Dataset not found"
- Verificar conexión a internet
- El dataset se descarga automáticamente de Hugging Face

### Error: "Model not found"
```bash
# Entrenar primero el modelo
python train_alzheimer_model.py
```

### Rendimiento bajo
- Aumentar número de épocas: `EPOCHS = 30`
- Ajustar learning rate: `LEARNING_RATE = 1e-4`
- Verificar que se use GPU si está disponible

## 📚 Dataset

**Fuente**: [Falah/Alzheimer_MRI](https://huggingface.co/datasets/Falah/Alzheimer_MRI) en Hugging Face

**Características**:
- ~6,400 imágenes MRI del cerebro
- 4 clases de demencia
- Imágenes en escala de grises
- Diferentes resoluciones (redimensionadas a 224×224)

**Distribución típica**:
- NonDemented: ~52%
- VeryMildDemented: ~28% 
- MildDemented: ~14%
- ModerateDemented: ~6%

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Para contribuir:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🙏 Agradecimientos

- **Dataset**: Falah/Alzheimer_MRI en Hugging Face
- **Modelo base**: ResNet18 de torchvision
- **Framework**: PyTorch y Hugging Face Datasets
- **Inspiración**: Investigación en detección temprana de Alzheimer

## 📞 Contacto

Para preguntas o sugerencias:
- 📧 Email: [tu-email@ejemplo.com]
- 🐙 GitHub: [tu-usuario]
- 💼 LinkedIn: [tu-perfil]

---

**⚠️ Nota Médica**: Este modelo es solo para fines educativos y de investigación. No debe usarse para diagnósticos médicos reales sin supervisión profesional.