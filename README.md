 # Install Transformers on Ubuntu 22.04 with CUDA

 Under develop!!!!!!!!!!!

 GPU is NVIDIA GeForce 940MX
 
 ## Refrences
 
 https://huggingface.co
 https://www.tensorflow.org/install/pip
 https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-723/install-guide/index.html
 
## Create a new environment
 
    python3 -m venv .env;
 
## Activate the environment

    source .env/bin/activate
 
## Install modules using PyTorch 

    python3 -m pip install --upgrade setuptools pip;
    python3 -m pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    python3 -m pip install nvidia-pyindex;
    python3 -m pip install --upgrade nvidia-tensorrt;

    pip3 install datasets;
    pip3 install evaluate;
    pip3 install tf-keras

    pip3 install transformers[torch]
    pip3 install accelerate
    pip3 install tf-keras;
 
 ### Verify the installation of PyTorch

 python -c "import torch; print(f'CUDA available? {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0)}')";

 python -c "from transformers import pipeline; print(pipeline('sentiment-analysis',device='cuda')('I love you'))";

 python -c "from transformers import pipeline; print(pipeline('sentiment-analysis',device='cuda')('I love you'))";
 python -c "from transformers import pipeline; summarizer = pipeline('sentiment-analysis',device='cuda'); outputs = summarizer('I love you'); print(outputs);"
 
<p>print(outputs[0]["summary_text"])


 ## Install modules using TensorFlow

 pip3 install tensorflow[and-cuda];
 pip3 install https://storage.googleapis.com/tensorflow/versions/2.16.1/tensorflow-2.16.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl;
 pip3 install https://storage.googleapis.com/tensorflow/versions/2.16.1/tensorflow_cpu-2.16.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl;

### Verify the installation of TensorFlow:
 python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))";
 python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))";

 # General information
 CUDA® is a parallel computing platform and programming model developed by NVIDIA for general computing on graphical processing units (GPUs). With CUDA, developers are able to dramatically speed up computing applications by harnessing the power of GPUs.


