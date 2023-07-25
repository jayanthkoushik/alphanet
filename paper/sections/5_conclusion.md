# Conclusion {#sec:conclusion}

The long-tailed nature of the world presents a challenge for
classification models, due to the imbalance in the number of training
samples per class. To address this problem, a number of methods have
been proposed, but the focus is generally on achieving the highest
overall accuracy. Consequently, many long-tail methods tend to have high
overall accuracy, but with unbalanced per-class accuracies where
frequent classes are learned well with high accuracies, and rare classes
are learned poorly with low accuracies. Such models can lead to biased
outcomes, which raises serious ethical concerns. In this paper, we
proposed AlphaNet, a rapid post hoc correction method that can be
applied to any model. Our simple method greatly improves the accuracy
for data-poor classes, and re-balances per-class classification
accuracies while preserving overall accuracy. AlphaNet can be deployed
in any application where the base classifiers cannot be changed, but
balanced performance is desirable -- thereby making it useful in
contexts where ethics, privacy, or intellectual property are concerns.
