# Conclusion {#sec:conclusion}

The long-tailed nature of the world presents a challenge for
classification models. Many long-tail methods tend to have high overall
accuracy, but with unbalanced per-class accuracies where frequent
classes are learned well with high accuracies and rare classes are
learned poorly with low accuracies. As such, the long-tailed world
represents one source of potential bias. To address this problem,
typical approaches resort to re-sampling or re-weighting of rare classes
but still focus on achieving the highest overall accuracy. Consequently,
these methods continue to suffer from low accuracy for data-poor
classes, and an accuracy imbalance across classes. Our method,
AlphaNet, provides a rapid post hoc correction that can be applied on
top of any model. Our simple method greatly improves the accuracy for
data-poor classes, and re-balances classification accuracy while
preserving overall accuracy. AlphaNet is deployable in any application
where the base classifiers cannot be changed, but balanced performance
is desirable -- thereby making it useful in contexts where ethics,
privacy, or intellectual property are concerns.
