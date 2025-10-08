#!/bin/bash

# Script de configuración automática del entorno virtual
# Para el proyecto de detección de Alzheimer

echo "🧠 CONFIGURACIÓN AUTOMÁTICA - DETECCIÓN DE ALZHEIMER"
echo "=================================================="

# Verificar que Python 3 está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 no está instalado. Por favor instálalo primero."
    exit 1
fi

echo "✅ Python 3 encontrado: $(python3 --version)"

# Crear entorno virtual
echo ""
echo "📦 Creando entorno virtual..."
python3 -m venv alzheimer_env

if [ $? -eq 0 ]; then
    echo "✅ Entorno virtual creado: alzheimer_env/"
else
    echo "❌ Error creando entorno virtual"
    exit 1
fi

# Activar entorno virtual
echo ""
echo "🔄 Activando entorno virtual..."
source alzheimer_env/bin/activate

# Verificar activación
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Entorno virtual activado: $VIRTUAL_ENV"
else
    echo "❌ Error activando entorno virtual"
    exit 1
fi

# Actualizar pip
echo ""
echo "⬆️  Actualizando pip..."
pip install --upgrade pip

# Instalar PyTorch (CPU version)
echo ""
echo "🔥 Instalando PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instalar otras dependencias
echo ""
echo "📚 Instalando dependencias del proyecto..."
pip install datasets matplotlib seaborn scikit-learn pandas tqdm

# Verificar instalación
echo ""
echo "🔍 Verificando instalación..."
python3 verify_setup.py

echo ""
echo "=================================================="
echo "🎉 ¡CONFIGURACIÓN COMPLETADA!"
echo "=================================================="
echo ""
echo "Para usar el proyecto:"
echo "1. Activar entorno: source alzheimer_env/bin/activate"
echo "2. Ejecutar proyecto: python3 train_alzheimer_model.py"
echo "3. Desactivar cuando termines: deactivate"
echo ""
echo "📁 Archivos del entorno virtual están en: alzheimer_env/"
echo "=================================================="