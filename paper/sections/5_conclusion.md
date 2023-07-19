# Conclusion {#sec:conclusion}

The long-tailed nature of the world presents a challenge for any model
that depends on learning from specific examples. Many long-tail
methods tend to have high overall accuracy, but with unbalanced
accuracies where frequent classes are learned well with high accuracies
and rare classes are learned poorly with low accuracies. As such, the
long-tailed world represents one source of potential bias. To address
this problem, typical approaches resort to re-sampling or re-weighting
of rare classes but still focus on achieving the highest overall
accuracy. Consequently, these methods continue to suffer from low
accuracy for data-poor classes, and an accuracy imbalance across
data-rich and data-poor classes. In contrast, our method, AlphaNet,
provides a rapid post hoc correction that can sit on top of any model
using classifiers. This simple method greatly improves the accuracy for
data-poor classes, and re-balances classification accuracy in a way that
overall classification accuracy is preserved. AlphaNet is deployable in
any application where the base classifiers cannot be changed, but
balanced performance is desirable -- thereby making it useful in
contexts where ethics, privacy, or intellectual property are concerns.
