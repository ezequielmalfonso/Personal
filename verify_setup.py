#!/usr/bin/env python3
"""
Script de verificación del entorno y configuración
Verifica que todas las dependencias estén instaladas correctamente
"""

import sys
import importlib
import subprocess
import os


def check_python_version():
    """Verifica la versión de Python"""
    print("🐍 Verificando versión de Python...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Se requiere Python 3.8+")
        return False


def check_dependencies():
    """Verifica que todas las dependencias estén instaladas"""
    print("\n📦 Verificando dependencias...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('datasets', 'Hugging Face Datasets'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('sklearn', 'Scikit-learn'),
        ('pandas', 'Pandas'),
        ('numpy', 'NumPy'),
        ('tqdm', 'TQDM'),
        ('PIL', 'Pillow')
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name} - OK")
        except ImportError:
            print(f"❌ {name} - NO ENCONTRADO")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Paquetes faltantes: {', '.join(missing_packages)}")
        print("Instalar con: pip install -r requirements.txt")
        return False
    
    return True


def check_cuda():
    """Verifica disponibilidad de CUDA"""
    print("\n🖥️  Verificando CUDA...")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA disponible - {device_count} GPU(s)")
            print(f"   GPU principal: {device_name}")
            return True
        else:
            print("⚠️  CUDA no disponible - Se usará CPU")
            print("   El entrenamiento será más lento pero funcional")
            return True
    except Exception as e:
        print(f"❌ Error verificando CUDA: {e}")
        return False


def check_internet_connection():
    """Verifica conexión a internet para descargar dataset"""
    print("\n🌐 Verificando conexión a internet...")
    try:
        import urllib.request
        urllib.request.urlopen('https://huggingface.co', timeout=10)
        print("✅ Conexión a Hugging Face - OK")
        return True
    except Exception:
        print("❌ Sin conexión a internet")
        print("   Se requiere internet para descargar el dataset")
        return False


def check_disk_space():
    """Verifica espacio en disco disponible"""
    print("\n💾 Verificando espacio en disco...")
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (1024**3)
        
        if free_gb >= 5:
            print(f"✅ Espacio libre: {free_gb} GB - OK")
            return True
        else:
            print(f"⚠️  Espacio libre: {free_gb} GB - Se recomiendan al menos 5 GB")
            return True
    except Exception as e:
        print(f"⚠️  No se pudo verificar espacio en disco: {e}")
        return True


def check_project_structure():
    """Verifica la estructura del proyecto"""
    print("\n📁 Verificando estructura del proyecto...")
    
    required_files = [
        'requirements.txt',
        'train_alzheimer_model.py',
        'demo_inference.py',
        'presentation_demo.py',
        'src/config.py',
        'src/data_utils.py',
        'src/model.py',
        'src/evaluation.py',
        'src/inference.py'
    ]
    
    required_dirs = [
        'src',
        'models',
        'results',
        'logs'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - NO ENCONTRADO")
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - NO ENCONTRADO")
            missing_dirs.append(dir_path)
    
    if missing_files or missing_dirs:
        print(f"\n⚠️  Archivos/directorios faltantes:")
        for item in missing_files + missing_dirs:
            print(f"   - {item}")
        return False
    
    return True


def test_basic_imports():
    """Prueba importaciones básicas del proyecto"""
    print("\n🧪 Probando importaciones del proyecto...")
    
    try:
        sys.path.append('src')
        
        # Probar importaciones
        from config import DEVICE, EPOCHS, BATCH_SIZE
        print(f"✅ Configuración cargada - Dispositivo: {DEVICE}")
        
        from data_utils import get_transforms
        train_transform, eval_transform = get_transforms()
        print("✅ Transformaciones de datos - OK")
        
        from model import create_model
        model = create_model(num_classes=4)
        print("✅ Creación de modelo - OK")
        
        print("✅ Todas las importaciones funcionan correctamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en importaciones: {e}")
        return False


def run_quick_test():
    """Ejecuta una prueba rápida del sistema"""
    print("\n🚀 Ejecutando prueba rápida...")
    
    try:
        import torch
        from torchvision import transforms
        from PIL import Image
        import numpy as np
        
        # Crear imagen de prueba
        test_image = Image.fromarray(np.random.randint(0, 255, (256, 256), dtype=np.uint8), mode='L')
        
        # Probar transformaciones
        sys.path.append('src')
        from data_utils import get_transforms
        train_transform, eval_transform = get_transforms()
        
        transformed = eval_transform(test_image)
        print(f"✅ Transformación de imagen - Shape: {transformed.shape}")
        
        # Probar modelo
        from model import create_model
        model = create_model(num_classes=4)
        model.eval()
        
        with torch.no_grad():
            output = model(transformed.unsqueeze(0))
            probabilities = torch.softmax(output, dim=1)
        
        print(f"✅ Inferencia de modelo - Output shape: {output.shape}")
        print("✅ Prueba rápida completada exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba rápida: {e}")
        return False


def main():
    """Función principal de verificación"""
    print("🔍 VERIFICACIÓN DEL ENTORNO")
    print("=" * 50)
    
    checks = [
        ("Versión de Python", check_python_version),
        ("Dependencias", check_dependencies),
        ("CUDA/GPU", check_cuda),
        ("Conexión a Internet", check_internet_connection),
        ("Espacio en Disco", check_disk_space),
        ("Estructura del Proyecto", check_project_structure),
        ("Importaciones", test_basic_imports),
        ("Prueba Rápida", run_quick_test)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error en {check_name}: {e}")
            results.append((check_name, False))
    
    # Resumen final
    print("\n" + "=" * 50)
    print("📋 RESUMEN DE VERIFICACIÓN")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {check_name}")
        if result:
            passed += 1
    
    print(f"\nResultado: {passed}/{total} verificaciones pasaron")
    
    if passed == total:
        print("\n🎉 ¡Todo está configurado correctamente!")
        print("Puedes ejecutar: python train_alzheimer_model.py")
    elif passed >= total - 2:
        print("\n⚠️  Configuración mayormente correcta")
        print("Puedes intentar ejecutar el entrenamiento")
    else:
        print("\n❌ Se encontraron problemas importantes")
        print("Revisa los errores antes de continuar")
    
    print("=" * 50)


if __name__ == "__main__":
    main()