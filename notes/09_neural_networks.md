# Neural Network Basics

## Layers and activations
A neural network stacks layers. Each layer computes a weighted sum and applies a nonlinear activation such as ReLU(z) = max(0, z). Without nonlinear activations, a stack of layers collapses into one linear model.

## Backpropagation
Backpropagation applies the chain rule to compute the gradient of the loss with respect to every weight, layer by layer from the output back to the input. Gradient descent then updates the weights.

## Softmax
For multi-class output, softmax turns scores into probabilities: softmax(z_i) = e^(z_i) / sum_j e^(z_j). The outputs are positive and sum to 1.

## Dropout
Dropout randomly zeroes a fraction of activations during training, which discourages co-adaptation and reduces overfitting. It is turned off at inference time.
