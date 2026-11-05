# Logistic Regression

## The model
Logistic regression is a classification model. It computes z = w · x + b and passes it through the sigmoid function sigma(z) = 1 / (1 + e^(-z)) to get a probability between 0 and 1.

## Decision threshold
By default we predict class 1 when the probability is at least 0.5. Moving the threshold trades precision against recall.

## Loss
Logistic regression is trained with binary cross-entropy (log loss): -[y log(p) + (1 - y) log(1 - p)]. Unlike MSE for this model, log loss is convex, so gradient descent finds the global minimum.

## Worked example
sigma(0) = 0.5. sigma(2) is about 0.881.
