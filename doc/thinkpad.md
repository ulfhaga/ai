# INSTALLATION

Computer Lenovo Thinkpad model T460p.
GPU is NVIDIA GeForce 940MX
During the installation of Ubuntu the drivers could be found.
Try first with the verify commands.

    \# List drivers
    <code>
    sudo ubuntu-drivers list --gpgpu;
    </code>

    \# If driver nvidia-driver-550 is missing do:  
    <code>
    sudo ubuntu-drivers install --gpgpu nvidia:550;
    </code>
    
    \# Install CUDA
    <code>
    sudo apt install nvidia-cuda-toolkit; 
    nvcc --version ;
    </code>
    
    \# NVIDIA cuDNN is a GPU-accelerated library of primitives for deep neural networks.
    <code>
    sudo apt install cudnn;
    </code>

    \# Verify. Give result nvidia-driver-550
    <code>
    nvidia-detector
    </code>

    \# Verify
    <code>
    nvidia-smi
    </code>


-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.120                Driver Version: 550.120        CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce 940MX           Off |   00000000:02:00.0 Off |                  N/A |
| N/A   35C    P8             N/A /  200W |       3MiB /   2048MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      1265      G   /usr/bin/gnome-shell                            0MiB |
+-----------------------------------------------------------------------------------------+


# Monitoring
    sudo apt install nvtop;
    nvtop;


# Reference

[NVIDIA drivers installation](https://documentation.ubuntu.com/server/how-to/graphics/install-nvidia-drivers/?_gl=1*1pbrw85*_gcl_au*NDcwOTczODIwLjE3MzQ1Mjk5NTM.&_ga=2.268301256.799803517.1734529952-1972945449.1734529952) 