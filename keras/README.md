# MNIST-keras example Using IBM Distributed Deep Learning

## About

The `mnist-tf-keras.py` script is a simple example that demonstrates how to modify a TensorFlow keras script to use IBM Distributed Deep Learning (DDL).

## Running MNIST-keras

Before running the either of the scripts, make sure there is a conda environment activated with the `ddl-tensorflow` package installed.

The examples should also be copied into a user directory by running the following command:

```bash
$ ddl-tensorflow-install-samples <somedir>
```

To run `mnist-tf-keras.py` on two machines named host1 and host2 that each have 4 GPUs:

```bash
$ ddlrun -H host1,host2 python mnist-tf-keras.py
```

To train the MNIST Keras model in TensorFlow's eager execution:

```bash
$ ddlrun -H host1,host2 python mnist-tf-keras.py --eager
```

&copy; Copyright IBM Corporation 2018, 2019
