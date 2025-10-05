# 🧠 Detección de Alzheimer mediante Deep Learning

## Análisis de Tomografías Cerebrales para Clasificación Automática

Este proyecto implementa un estudio completo de machine learning para la detección automática de diferentes niveles de Alzheimer a partir de tomografías cerebrales, utilizando técnicas de deep learning y análisis estadístico riguroso.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Tabla de Contenidos

- [Características](#-características)
- [Instalación](#-instalación)
- [Uso Rápido](#-uso-rápido)
- [Estructura del Proyecto](#-estructura-del-proyecto)
- [Metodología](#-metodología)
- [Resultados](#-resultados)
- [Contribuir](#-contribuir)
- [Licencia](#-licencia)

## ✨ Características

### 🔬 **Análisis Completo**
- **Análisis exploratorio exhaustivo** con visualizaciones interactivas
- **Análisis de calidad de imágenes** y detección de anomalías
- **Estadísticas descriptivas** y distribuciones de clases

### 🏗️ **Múltiples Arquitecturas**
- **ResNet18/50**: Redes residuales preentrenadas
- **EfficientNet-B0**: Arquitectura eficiente y moderna
- **CNN Personalizada**: Diseñada específicamente para el problema
- **Comparación sistemática** de rendimiento

### 📊 **Evaluación Rigurosa**
- **Métricas completas**: Accuracy, F1-score, Precision, Recall, AUC
- **Validación cruzada estratificada** (k-fold)
- **Matrices de confusión** y curvas ROC/PR
- **Análisis por clase** detallado

### 🎯 **Preprocesamiento Avanzado**
- **Aumentaciones de datos** específicas para imágenes médicas
- **Normalización tipo ImageNet** adaptada
- **Balanceo de clases** automático
- **Pipeline reproducible** y configurable

### 📈 **Visualizaciones Profesionales**
- **Gráficos de alta calidad** para presentaciones
- **Historial de entrenamiento** interactivo
- **Comparaciones entre modelos** visuales
- **Reportes automáticos** en formato Markdown

## 🚀 Instalación

### Prerrequisitos
- Python 3.8 o superior
- CUDA 11.0+ (opcional, para GPU)

### Instalación Rápida

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/alzheimer-detection.git
cd alzheimer-detection

# Crear entorno virtual
python -m venv alzheimer_env
source alzheimer_env/bin/activate  # Linux/Mac
# alzheimer_env\\Scripts\\activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Instalación para Desarrollo

```bash
# Instalar dependencias de desarrollo
pip install -r requirements.txt
pip install -e .

# Configurar pre-commit hooks
pre-commit install
```

## 🎯 Uso Rápido

### Ejecución Completa del Estudio

```bash
# Estudio completo (20 épocas, múltiples modelos)
python run_complete_study.py --epochs 20 --models resnet18 resnet50 custom_cnn

# Con configuración personalizada
python run_complete_study.py \\
    --epochs 30 \\
    --batch-size 64 \\
    --models resnet18 resnet50 efficientnet_b0 \\
    --skip-analysis  # Saltar análisis exploratorio
```

### Prueba Rápida

```bash
# Prueba rápida (3 épocas, solo ResNet18)
python run_complete_study.py --quick-test
```

### Uso de Módulos Individuales

```python
from alzheimer_detection_study import AlzheimerDatasetAnalyzer
from models_and_training import AlzheimerClassifier

# Análisis exploratorio
analyzer = AlzheimerDatasetAnalyzer()
analyzer.load_dataset()
analyzer.exploratory_analysis()

# Entrenar modelo específico
classifier = AlzheimerClassifier(num_classes=4, model_name='resnet18')
model = classifier.build_model()
```

### Notebook Interactivo

```bash
# Abrir notebook de presentación
jupyter notebook Alzheimer_Detection_Presentation.ipynb
```

## 📁 Estructura del Proyecto

```
alzheimer-detection/
├── 📄 README.md                              # Este archivo
├── 📄 requirements.txt                       # Dependencias
├── 📄 run_complete_study.py                  # Script principal
├── 📄 alzheimer_detection_study.py           # Análisis exploratorio
├── 📄 models_and_training.py                 # Modelos y entrenamiento
├── 📓 Alzheimer_Detection_Presentation.ipynb # Notebook de presentación
├── 📁 results/                               # Resultados del estudio
│   ├── 📄 alzheimer_study_results.json       # Resultados en JSON
│   ├── 📄 models_comparison.csv              # Comparación de modelos
│   └── 📄 REPORTE_FINAL.md                   # Reporte final
├── 📁 figures/                               # Visualizaciones
│   ├── 🖼️ exploratory_analysis.png           # Análisis exploratorio
│   ├── 🖼️ confusion_matrix_*.png             # Matrices de confusión
│   ├── 🖼️ roc_curves_*.png                   # Curvas ROC
│   └── 🖼️ training_history_*.png             # Historial de entrenamiento
├── 📁 models/                                # Modelos entrenados
│   ├── 🤖 best_model_resnet18.pt             # Mejor ResNet18
│   ├── 🤖 best_model_resnet50.pt             # Mejor ResNet50
│   └── 🤖 best_model_custom_cnn.pt           # Mejor CNN personalizada
└── 📁 docs/                                  # Documentación adicional
    ├── 📄 methodology.md                     # Metodología detallada
    └── 📄 clinical_validation.md             # Guía de validación clínica
```

## 🔬 Metodología

### Dataset
- **Fuente**: [Falah/Alzheimer_MRI](https://huggingface.co/datasets/Falah/Alzheimer_MRI) (HuggingFace)
- **Clases**: No Dementia, Very Mild Dementia, Mild Dementia, Moderate Dementia
- **Tipo**: Imágenes de resonancia magnética cerebral
- **División**: 85% entrenamiento, 15% validación, test independiente

### Preprocesamiento
1. **Conversión**: Escala de grises → 3 canales RGB
2. **Redimensionado**: 224×224 píxeles (compatible con ImageNet)
3. **Aumentaciones** (solo entrenamiento):
   - Rotación aleatoria: ±10°
   - Traslación: ±5% en X,Y
   - Ajuste de brillo/contraste: ±10%
   - Ruido gaussiano suave (σ=0.02)
4. **Normalización**: Media y desviación estándar de ImageNet

### Modelos Evaluados
- **ResNet18**: Red residual ligera, rápida convergencia
- **ResNet50**: Red residual profunda, mayor capacidad
- **EfficientNet-B0**: Arquitectura eficiente y moderna
- **CNN Personalizada**: Diseñada específicamente para el problema

### Entrenamiento
- **Optimizador**: AdamW (lr=3e-4, weight_decay=1e-4)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Loss**: CrossEntropyLoss con pesos de clase
- **Regularización**: Dropout, BatchNorm, Early Stopping
- **Mixed Precision**: Habilitado en GPU para eficiencia

### Evaluación
- **Métricas**: Accuracy, F1-macro/weighted, Precision, Recall, AUC
- **Validación cruzada**: 5-fold estratificada
- **Visualizaciones**: Matrices de confusión, curvas ROC/PR
- **Interpretabilidad**: Análisis de características importantes

## 📊 Resultados

### Rendimiento de Modelos (Ejemplo)

| Modelo | Accuracy | F1-Macro | F1-Weighted | Precision | Recall | AUC-Macro |
|--------|----------|----------|-------------|-----------|--------|-----------|
| ResNet50 | 0.8721 | 0.8456 | 0.8634 | 0.8234 | 0.8567 | 0.9123 |
| ResNet18 | 0.8542 | 0.8234 | 0.8456 | 0.8123 | 0.8345 | 0.8967 |
| EfficientNet-B0 | 0.8634 | 0.8345 | 0.8523 | 0.8234 | 0.8456 | 0.9045 |
| Custom CNN | 0.8123 | 0.7891 | 0.8034 | 0.7756 | 0.8123 | 0.8734 |

### Hallazgos Clave
- ✅ **Las redes preentrenadas superan significativamente a las CNNs desde cero**
- ✅ **ResNet50 muestra el mejor balance entre precisión y robustez**
- ✅ **El preprocesamiento con aumentaciones mejora la generalización**
- ✅ **Los pesos de clase son efectivos para manejar el desbalance**
- ⚠️ **Se requiere validación externa para aplicación clínica**

### Archivos Generados
- **Modelos entrenados**: `models/best_model_*.pt`
- **Resultados detallados**: `results/alzheimer_study_results.json`
- **Reporte final**: `results/REPORTE_FINAL.md`
- **Visualizaciones**: `figures/*.png`

## 🎯 Casos de Uso

### 🏥 **Aplicación Clínica**
- **Apoyo al diagnóstico**: Segunda opinión para radiólogos
- **Screening masivo**: Detección temprana en poblaciones de riesgo
- **Seguimiento longitudinal**: Monitoreo de progresión
- **Telemedicina**: Diagnóstico remoto en áreas rurales

### 🔬 **Investigación**
- **Estudios epidemiológicos**: Análisis de grandes cohortes
- **Desarrollo de fármacos**: Endpoint objetivo en ensayos clínicos
- **Biomarcadores**: Identificación de patrones de neuroimagen
- **Medicina personalizada**: Estratificación de pacientes

### 🎓 **Educación**
- **Formación médica**: Herramienta de enseñanza interactiva
- **Simulación clínica**: Casos de práctica para residentes
- **Investigación académica**: Plataforma para estudios de ML médico

## 🛠️ Configuración Avanzada

### Variables de Entorno

```bash
# Configuración opcional
export CUDA_VISIBLE_DEVICES=0,1  # GPUs a usar
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Optimización de memoria
export WANDB_PROJECT="alzheimer-detection"  # Logging con Weights & Biases
```

### Configuración de Hiperparámetros

```python
# Personalizar configuración
config = {
    'batch_size': 64,
    'learning_rate': 3e-4,
    'epochs': 30,
    'weight_decay': 1e-4,
    'dropout_rate': 0.5,
    'augmentation_strength': 0.8
}
```

### Optimización de Hiperparámetros

```bash
# Usando Optuna para búsqueda automática
python optimize_hyperparameters.py --trials 100 --model resnet18
```

## 🧪 Testing

### Ejecutar Tests

```bash
# Tests unitarios
pytest tests/ -v

# Tests de integración
pytest tests/integration/ -v

# Coverage
pytest --cov=alzheimer_detection tests/
```

### Validación de Modelos

```bash
# Validar modelo entrenado
python validate_model.py --model models/best_model_resnet18.pt --test-data path/to/test
```

## 📚 Documentación Adicional

### Guías Detalladas
- [📖 Metodología Completa](docs/methodology.md)
- [🏥 Validación Clínica](docs/clinical_validation.md)
- [🔧 Guía de Desarrollo](docs/development_guide.md)
- [🚀 Despliegue en Producción](docs/deployment.md)

### Tutoriales
- [🎓 Tutorial Básico](tutorials/basic_usage.ipynb)
- [🔬 Análisis Avanzado](tutorials/advanced_analysis.ipynb)
- [🏗️ Modelos Personalizados](tutorials/custom_models.ipynb)

### API Reference
- [📋 Documentación de API](https://alzheimer-detection.readthedocs.io/)

## 🤝 Contribuir

¡Las contribuciones son bienvenidas! Por favor, lee nuestra [guía de contribución](CONTRIBUTING.md).

### Formas de Contribuir
- 🐛 **Reportar bugs** y problemas
- 💡 **Sugerir nuevas características**
- 📝 **Mejorar documentación**
- 🧪 **Añadir tests**
- 🏗️ **Implementar nuevas arquitecturas**

### Proceso de Desarrollo

```bash
# 1. Fork del repositorio
# 2. Crear rama para feature
git checkout -b feature/nueva-caracteristica

# 3. Hacer cambios y commits
git commit -m "feat: añadir nueva característica"

# 4. Push y crear Pull Request
git push origin feature/nueva-caracteristica
```

## 📄 Licencia

Este proyecto está licenciado bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## 🙏 Agradecimientos

- **Dataset**: [Falah/Alzheimer_MRI](https://huggingface.co/datasets/Falah/Alzheimer_MRI)
- **PyTorch Team**: Por el framework de deep learning
- **HuggingFace**: Por la plataforma de datasets
- **Comunidad científica**: Por la investigación en IA médica

## 📞 Contacto y Soporte

- **Issues**: [GitHub Issues](https://github.com/tu-usuario/alzheimer-detection/issues)
- **Discusiones**: [GitHub Discussions](https://github.com/tu-usuario/alzheimer-detection/discussions)
- **Email**: tu-email@ejemplo.com

## ⚠️ Disclaimer Médico

**IMPORTANTE**: Este software es solo para fines de investigación y educación. No está destinado para uso clínico directo. Cualquier aplicación médica requiere:

- ✅ Validación clínica rigurosa
- ✅ Aprobación regulatoria (FDA, CE, etc.)
- ✅ Supervisión médica profesional
- ✅ Cumplimiento de normativas locales

**No usar como sustituto del juicio clínico profesional.**

---

<div align="center">

**🧠 Detección de Alzheimer mediante Deep Learning**

*Contribuyendo al futuro del diagnóstico médico asistido por IA*

[![Stars](https://img.shields.io/github/stars/tu-usuario/alzheimer-detection?style=social)](https://github.com/tu-usuario/alzheimer-detection/stargazers)
[![Forks](https://img.shields.io/github/forks/tu-usuario/alzheimer-detection?style=social)](https://github.com/tu-usuario/alzheimer-detection/network/members)

</div>