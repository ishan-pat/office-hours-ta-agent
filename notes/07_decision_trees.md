# Decision Trees and Ensembles

## Splitting
A decision tree splits the data on the feature and threshold that most reduce impurity. Common impurity measures are Gini impurity and entropy.

## Gini impurity
Gini = 1 - sum over classes of p_k^2. A pure node has Gini 0. For two classes split 50/50, Gini is 0.5.

## Overfitting in trees
Deep trees memorize the training data. Limit max depth, require a minimum number of samples per leaf, or prune.

## Random forests
A random forest trains many trees on bootstrap samples, with each split considering a random subset of features, and averages their votes. This lowers variance compared with a single tree.

## Gradient boosting
Boosting trains trees one after another, each fitting the errors of the ones before. XGBoost and LightGBM are popular implementations.
