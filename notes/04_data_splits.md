# Train, Validation and Test Splits

## Why split
We train on the training set, tune hyperparameters on the validation set, and report final performance once on the test set. Using the test set for tuning leaks information and makes the reported score optimistic.

## Typical ratios
A common split is 70/15/15 or 80/10/10. With small datasets, k-fold cross-validation is preferred.

## k-fold cross-validation
Split the data into k folds. Train on k-1 folds and validate on the remaining one, rotating k times, then average the k scores. k = 5 or k = 10 are common choices.

## Data leakage
Leakage happens when information from outside the training data reaches the model, for example fitting a scaler on the full dataset before splitting. Always fit preprocessing on the training set only.
