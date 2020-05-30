
This is a Convolutional Neural Network (CNN) and Deep Neural Network (DNN) package developed 
for classification of calorimeter signals.

This package directly employed backpropagation algorithm, and was built from the ground,
under the CentOS 7.8 enviroment using gcc 4.8.
It does not rely on any availabe external Neural Network libraries, 
and is purely for research purpose. 

The Matrix operation has also been implemented independently in the package. 
One can also turn on parallel computing for Matrix operation.
One can also turn on parallel computing for batch level traning.

One can use this libary to construct CNN, DNN or mixed neural networks.
With slight modification, one can also use it to build auto-encoder NNs.

Some techniques used for mitigate overfitting has also been implemented,
such as Drop-out, L2/L1 regularization.

In this library, NN layers can be divided into two categories: 
1-Dimensional (1D) layer and 2D layer.

For example, fully connected layer is 1-Dimensional, since its neurons are arranged in a collum
vector; Convolutional layer is 2-Dimensional, since the kernels and the output image of the 
convolutional layer are all in 2D matrix form. Pooling layer is also 2D.

Currently, in this package, 2D layer CAN BE followed by 2D and 1D layers, 
for example: 
{Image->CNN->Pooling->CNN->FC->FC} is allowed. 
However, 1D layer CANNOT BE followd by 2D layer (only CAN BE followed by 1D layer),
for example: 
{FC->CNN} type of connection is not implemented, 
because application of this type of connection is not encountered 
in calorimeter signal classification.


email xb4zp@virginia.edu for more details.
