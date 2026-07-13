#!/bin/bash

echo "========================================"
echo " INFORMACIÓN DEL SISTEMA"
echo "========================================"
echo "Usuario: $(whoami)"
echo "Sistema:"
lsb_release -a 2>/dev/null || cat /etc/os-release

echo ""
echo "Kernel:"
uname -r

echo ""
echo "Arquitectura:"
uname -m

echo ""
echo "========================================"
echo " PYTHON Y PIP"
echo "========================================"
echo "Python 3:"
python3 --version 2>/dev/null || echo "Python3 no encontrado"

echo ""
echo "Pip:"
pip3 --version 2>/dev/null || echo "pip3 no encontrado"

echo ""
echo "========================================"
echo " NVIDIA GPU"
echo "========================================"
echo "nvidia-smi:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi no encontrado. Puede que el driver NVIDIA no esté instalado."
fi

echo ""
echo "========================================"
echo " DRIVER NVIDIA"
echo "========================================"
echo "Versión del driver:"
cat /proc/driver/nvidia/version 2>/dev/null || echo "No se pudo leer /proc/driver/nvidia/version"

echo ""
echo "========================================"
echo " CUDA"
echo "========================================"
echo "nvcc:"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "nvcc no encontrado. Puede que CUDA Toolkit no esté instalado."
fi

echo ""
echo "CUDA en /usr/local:"
ls -l /usr/local/ | grep cuda || echo "No se encontró carpeta CUDA en /usr/local"

echo ""
echo "Variables CUDA:"
echo "CUDA_HOME=$CUDA_HOME"
echo "PATH=$PATH" | grep -o "/usr/local/cuda[^:]*" || echo "No hay ruta CUDA visible en PATH"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo ""
echo "========================================"
echo " PYTORCH, SI YA ESTÁ INSTALADO"
echo "========================================"
python3 - << 'EOF'
try:
    import torch
    print("Torch:", torch.__version__)
    print("CUDA disponible:", torch.cuda.is_available())
    print("CUDA versión usada por PyTorch:", torch.version.cuda)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("Capacidad CUDA:", torch.cuda.get_device_capability(0))
        print("VRAM total GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
except Exception as e:
    print("PyTorch no instalado o error al cargarlo:")
    print(e)
EOF

echo ""
echo "========================================"
echo " FIN DEL DIAGNÓSTICO"
echo "========================================"
