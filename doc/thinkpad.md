# INSTALLATION

Computer Lenovo Thinkpad modell T460p.
GPU is NVIDIA GeForce 940MX

    # Install Phyton
    sudo apt-get install python3;
    sudo apt-get install python-is-python3;
    sudo apt-get install python3-libnvinfer-dev;

    # List drivers
    sudo ubuntu-drivers list --gpgpu;

    # Install drivers 
    sudo ubuntu-drivers install --gpgpu nvidia:560;
    sudo apt install nvidia-utils-560-server;
    
    # Install CUDA
    sudo apt install nvidia-cuda-toolkit; 
    nvcc --version ;
    # NVIDIA cuDNN is a GPU-accelerated library of primitives for deep neural networks.
    sudo apt install cudnn;

    # Verifiera
    nvidia-detector
    nvidia-smi