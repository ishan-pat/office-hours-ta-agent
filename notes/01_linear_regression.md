# Linear Regression

## The model
Linear regression predicts a continuous target y from features x with a weighted sum: y_hat = w · x + b. The weights w and bias b are the parameters we learn.

## Mean squared error
We measure fit with mean squared error (MSE): the average of (y - y_hat)^2 over all training examples. Squaring makes large errors count more than small ones and keeps the loss differentiable.

## Closed-form solution
For ordinary least squares there is a closed-form answer, the normal equation: w = (X^T X)^(-1) X^T y. It is exact but becomes slow when there are many features, because inverting X^T X costs roughly the cube of the number of features.

## R squared
R^2 = 1 - (sum of squared residuals / total sum of squares). An R^2 of 1 means a perfect fit; 0 means the model does no better than always predicting the mean of y.
