# INSTALLATION

Computer Lenovo Thinkpad model T460p.
GPU is NVIDIA GeForce 940MX
During the installation of Ubuntu the drivers could be found.
Try first with the verify commands.

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

    # Verify
    nvidia-detector
    nvidia-smi

    # Monitor
    sudo apt install nvtop;
    nvtop;