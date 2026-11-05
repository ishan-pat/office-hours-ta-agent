# Overfitting and Regularization

## Overfitting
A model overfits when it learns noise in the training data: training error is low but validation error is high. Underfitting is the opposite: both errors are high because the model is too simple.

## Bias-variance tradeoff
High bias means the model's assumptions are too strong (underfitting). High variance means predictions change a lot with different training sets (overfitting). Adding model complexity lowers bias and raises variance.

## L2 regularization (ridge)
Ridge adds lambda * sum(w^2) to the loss. It shrinks weights toward zero but rarely makes them exactly zero.

## L1 regularization (lasso)
Lasso adds lambda * sum(|w|) to the loss. It can push weights to exactly zero, so it also performs feature selection.

## Other remedies
More training data, early stopping, dropout (for neural networks) and simpler models all reduce overfitting.
