# INSTALLATION

Computer Tinkpad modell T460p.
GPU is NVIDIA GeForce 940MX

    sudo ubuntu-drivers list --gpgpu
    sudo ubuntu-drivers install  --gpgpu nvidia:560
    sudo apt install nvidia-utils-560-server
    sudo apt-get install python-is-python3
    sudo apt-get install python3-libnvinfer-dev;

    sudo apt install nvidia-cuda-toolkit  # 11.5.1
    nvcc --version 
    # NVIDIA cuDNN is a GPU-accelerated library of primitives for deep neural networks.
    sudo apt install cudnn

    nvidia-detector
    nvidia-smi