# MNIST example Using IBM Distributed Deep Learning

## About

The MNIST script is a simple example that demonstrates how to modify a TensorFlow script to use IBM Distributed Deep Learning (DDL). For details on the changes that were made to the `mnist.py` scipt, see: `$CONDA_PREFIX/doc/ddl-tensorflow/Tutorial.md`.

## Running MNIST

Before running the mnist scripts, make sure there is a conda environment activated with the `ddl-tensorflow` package installed.

The examples should also be copied into a user directory by running the following command:

```bash
$ ddl-tensorflow-install-samples <somedir>
```

### Original MNIST

The `mnist.py` script is a convolution network implementation that uses the MNIST databse of handwriiten digits. To run the original `mnist.py` script:

```bash
$ python mnist.py
```

### DDL MNIST Using DDL_OPTIONS

The `mnist-env.py` script modifies mnist.py in order to utilize DDL using DDL_OPTIONS. To run `mnist-env.py` on two machines named host1 and host2 that each have 4 GPUs:

```bash
$ ddlrun -H host1,host2 python mnist-env.py
```

### DDL MNIST With TensorFlow's eager execution

`mnist-eager-env.py` script is a version of the MNIST script running with TensorFlow's eager execution

```bash
$ ddlrun -H host1,host2 python mnist-eager-env.py
```

### DDL MNIST Without Using DDL_OPTIONS

The `mnist-init.py` script modifies `mnist.py` in order to utilize DDL without using the DDL_OPTIONS environment vairable. You have to pass the options using command line argument (--ddl_options). To run `mnist-init.py` on two machines named host1 and host2 that each have 4 GPUs:

```bash
ddlrun --no_ddloptions -H host1,host2 python mnist-init.py --ddl_options="-mode b:4x2"
```

&copy; Copyright IBM Corporation 2018, 2019
