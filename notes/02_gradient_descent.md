# Gradient Descent

## The idea
Gradient descent minimizes a loss by repeatedly stepping opposite the gradient: w_new = w_old - learning_rate * gradient. The gradient points uphill, so its negative points downhill.

## Learning rate
The learning rate sets the step size. Too large and the loss oscillates or diverges; too small and training is very slow. A common first check when loss explodes is to lower the learning rate by a factor of 10.

## Batch, stochastic, mini-batch
Batch gradient descent uses every example per step. Stochastic gradient descent (SGD) uses one example per step, which is noisy but cheap. Mini-batch gradient descent uses a small batch (for example 32 or 64 examples) and is the default in practice.

## Worked example
For f(w) = (w - 3)^2 the gradient is 2(w - 3). Starting at w = 0 with learning rate 0.1, the first step gives w = 0 - 0.1 * 2 * (0 - 3) = 0.6.
