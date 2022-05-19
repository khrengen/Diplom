# Tensor Train for compressing neural networks

## What is it?
This is the low-rank decomposition of the tensor. Each dimension of the tensor corresponds to a set of matrices and any element of the tensor can be obtained by multiplying the corresponding matrices from this sets. For example, an element of a tensor with *4* dimensions in the tensor train format is calculated like this:

## what is it for?
This format is used instead of a matrix of weights in a fully connected layer of a neural network. This allows to significantly reduce the number of trainable parameters before training and train the layer with a new computational graph. This makes learning more stable and memory-efficient.

## Aplication
Required to copy the tt_linearV2.py module. Then just add a TT-layer to the PyTorch NN. Examples of use are in experiments.

## Hyperparameters
The layer has 3 hyperparameters:

in_modes - corresponds to the input of a fully connected layer but is factorized

out_modes - same for output

ranks - corresponds to the sizes of matrices in TT-format