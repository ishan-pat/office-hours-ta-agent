# k-Means Clustering

## The algorithm
k-means is unsupervised. Pick k initial centroids, assign each point to the nearest centroid, move each centroid to the mean of its assigned points, and repeat until assignments stop changing.

## Choosing k
The elbow method plots within-cluster sum of squares against k and looks for the bend. The silhouette score is another option; higher is better.

## Limitations
k-means assumes roughly round clusters of similar size, is sensitive to feature scale (standardize first), and depends on initialization. k-means++ initialization reduces bad starts.
