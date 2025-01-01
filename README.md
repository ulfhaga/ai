# Install Transformers on Ubuntu 22.04 with CUDA

## Prepare

OS is Ubuntu 22.04. Important is to use Nvidia CUDA toolkit to run on GPU.

[Installation](doc/thinkpad.md)

We are using PyTorch. It is an open-source machine learning library. 
PyTorch has built-in support for CUDA, NVIDIA's parallel computing platform, 
allowing for easy and efficient execution of tensor operations on GPUs, 
which significantly speeds up deep learning computations.

## Install Phyton

    sudo apt-get install python3;
    sudo apt-get install python-is-python3;
    sudo apt-get install python3.10-venv;

## Create a new environment
 
    python3 -m venv .env;
 
## Activate the environment

    source .env/bin/activate

In  Visual Studio Code.  Select Interpreter command from the Command Palette (Ctrl+Shift+P) and select 
the python interpreter that belongs to the new virtual environment.   
 
## Install modules using PyTorch 

    python3 -m pip install --upgrade setuptools pip;
    python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 ;
    python3 -m pip install nvidia-pyindex;
    python3 -m pip install --upgrade nvidia-tensorrt;

    pip3 install datasets;
    pip3 install evaluate;
    pip3 install tf-keras;

    pip3 install transformers[torch];
    pip3 install accelerate;
    pip3 install zstandard;
    pip3 install psutil;
    pip3 install requests;
    pip3 install jq;

    pip3 install -U "huggingface_hub[cli]"


 
 ### Verify the installation of PyTorch

    python -c "import torch; print(f'CUDA available? {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0)}')";

    python -c "from transformers import pipeline; summarizer = pipeline('sentiment-analysis',device='cuda'); outputs = summarizer('I love you'); print(outputs);"  

With the Linux command nvtop you can see the GPU status for NVIDIA GPUs.

## Install modules using TensorFlow 

 TensorFlow be used instead of PyTorch. I am not using that.  Wrote some instruction on the page [Install TensorFlow](doc/tensorflow.md)

## Examples

Will be found in folder examples. They are based of examples from https://huggingface.co/.


Examples to run:

    source .env/bin/activate;
    python examples/torchtest.py 2>/dev/null;
    python examples/textgen.py  2>/dev/null;
    python examples/text2gen.py  2>/dev/null;
    python examples/processingdata.py 2>/dev/null;
    python examples/training.py 2>/dev/null

    



## General information
 
CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs). With CUDA, developers are able to dramatically speed up computing applications by harnessing the power of GPUs.

PyTorch is an open-source machine learning library primarily used for deep learning and artificial intelligence (AI) applications. Developed by Facebook's AI Research lab (FAIR) and released in 2016, PyTorch has quickly become one of the most popular frameworks for building and training neural networks. 

## Refrences
 
 https://huggingface.co
 https://www.tensorflow.org/install/pip
 https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-723/install-guide/index.html
 https://developer.nvidia.com/cuda-toolkit
 
