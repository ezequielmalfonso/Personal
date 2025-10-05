#!/usr/bin/env python3
"""
Script de configuración automática del proyecto
Instala dependencias y configura el entorno
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Ejecuta un comando y maneja errores"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} - Completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Error:")
        print(f"   {e.stderr}")
        return False


def install_dependencies():
    """Instala las dependencias del proyecto"""
    print("📦 INSTALANDO DEPENDENCIAS")
    print("-" * 30)
    
    # Actualizar pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", 
                      "Actualizando pip"):
        return False
    
    # Instalar dependencias
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                      "Instalando dependencias"):
        return False
    
    return True


def create_directories():
    """Crea los directorios necesarios"""
    print("\n📁 CREANDO DIRECTORIOS")
    print("-" * 25)
    
    directories = ['models', 'results', 'logs', 'results/presentation']
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ {directory}/")
        except Exception as e:
            print(f"❌ Error creando {directory}/: {e}")
            return False
    
    return True


def verify_installation():
    """Verifica que la instalación sea correcta"""
    print("\n🔍 VERIFICANDO INSTALACIÓN")
    print("-" * 28)
    
    return run_command(f"{sys.executable} verify_setup.py", 
                      "Ejecutando verificación completa")


def main():
    """Función principal de configuración"""
    print("🚀 CONFIGURACIÓN AUTOMÁTICA DEL PROYECTO")
    print("=" * 45)
    print("Sistema de Detección de Alzheimer con ResNet18")
    print("=" * 45)
    
    steps = [
        ("Instalación de dependencias", install_dependencies),
        ("Creación de directorios", create_directories),
        ("Verificación final", verify_installation)
    ]
    
    for step_name, step_func in steps:
        if not step_func():
            print(f"\n❌ Error en: {step_name}")
            print("La configuración no se completó correctamente.")
            sys.exit(1)
    
    print("\n" + "=" * 45)
    print("🎉 ¡CONFIGURACIÓN COMPLETADA EXITOSAMENTE!")
    print("=" * 45)
    print("\n📚 Próximos pasos:")
    print("1. Entrenar el modelo:")
    print("   python train_alzheimer_model.py")
    print("\n2. Probar inferencia:")
    print("   python demo_inference.py")
    print("\n3. Generar presentación:")
    print("   python presentation_demo.py")
    print("\n📖 Para más información:")
    print("   Leer README.md")
    print("=" * 45)


if __name__ == "__main__":
    main()