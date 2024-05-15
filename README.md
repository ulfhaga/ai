 # Install Transformers on Ubuntu 22.04 with CUDA
 
 Just testing
 
 ## Refrences
 
 https://huggingface.co
 https://www.tensorflow.org/install/pip
 https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-723/install-guide/index.html
 
 python3 -m venv .env;
 
 source .env/bin/activate
 
 pip3 install torch torchvision torchaudio;
 
 python3 -m pip install --upgrade setuptools pip
 python3 -m pip install nvidia-pyindex
 python3 -m pip install --upgrade nvidia-tensorrt


 
 sudo apt-get install python3-libnvinfer-dev
 
 
 Install  tensorflow
 
 pip3 install https://storage.googleapis.com/tensorflow/versions/2.16.1/tensorflow-2.16.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl;
 pip3 install https://storage.googleapis.com/tensorflow/versions/2.16.1/tensorflow_cpu-2.16.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
 

 python3 -m pip install tensorflow[and-cuda]
 
 Verify the installation:
 python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
 python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

 pip install tf-keras;
 pip install transformers;
 python3 -c "from transformers import pipeline; print(pipeline('sentiment-analysis')('we love you'))"


