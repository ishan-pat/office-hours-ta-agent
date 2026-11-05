# Classification Metrics

## Confusion matrix
True positives (TP), false positives (FP), true negatives (TN) and false negatives (FN) summarize a binary classifier's predictions.

## Accuracy
Accuracy = (TP + TN) / total. It is misleading on imbalanced data: if 99% of emails are not spam, always predicting "not spam" scores 99%.

## Precision and recall
Precision = TP / (TP + FP): of the items we flagged, how many were right. Recall = TP / (TP + FN): of the true positives, how many we found.

## F1 score
F1 is the harmonic mean of precision and recall: F1 = 2 * P * R / (P + R). It is high only when both are high.

## ROC and AUC
The ROC curve plots true positive rate against false positive rate across thresholds. AUC is the area under it; 0.5 is random guessing and 1.0 is perfect ranking.
