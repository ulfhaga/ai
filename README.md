 https://huggingface.co/docs/transformers/main/en/installation
 
 python3 -m venv .env;
 
 source .env/bin/activate
 
 pip3 install torch torchvision torchaudio;
 
 python3 -m pip install --upgrade setuptools pip
 python3 -m pip install --upgrade nvidia-tensorrt
 python3 -m pip install nvidia-pyindex
 
 
 pip3 install https://storage.googleapis.com/tensorflow/versions/2.16.1/tensorflow-2.16.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl;
 
 pip3 install https://storage.googleapis.com/tensorflow/versions/2.16.1/tensorflow_cpu-2.16.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
 
 
 
 python3 -m pip install tensorflow[and-cuda]
 
# Verify the installation:
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

pip install tf-keras;
pip install transformers;
:wq

