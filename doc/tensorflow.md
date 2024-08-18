 # TensorFlow
 
 ## Install modules

 We are not using the TensorFlow as machine learning library.

    pip3 install tensorflow[and-cuda];
    pip3 install https://storage.googleapis.com/tensorflow/versions/2.16.1/tensorflow-2.16.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl;
    pip3 install https://storage.googleapis.com/tensorflow/versions/2.16.1/tensorflow_cpu-2.16.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl;

### Verify the installation 

 python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))";
 python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))";