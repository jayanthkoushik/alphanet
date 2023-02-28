# Conclusion {#sec:conclusion}

The long-tailed nature of the world presents a challenge for any model that
depends on learning over specific examples. Most long-tailed methods tend to
have high overall accuracy, but with unbalanced accuracies where frequent
classes are learned well with high accuracies and rare classes are learned
poorly with low accuracies. As such, the long-tailed world represents one source
of potential bias [along with the models themselves and system designer
decisions; @2022.Ferrante.Lara]. To address this problem, typical approaches
resort to re-sampling or re-weighting of rare classes but still focus on
achieving the highest overall accuracy. Consequently, these methods continue to
suffer from an accuracy imbalance across data-rich and data-poor classes. In
contrast, our method, AlphaNet, provides a rapid 5 minute _post-hoc_ correction
that can sit on top of any model using classifiers. This simple method
re-balances classification accuracy quickly so that rare classes have much
higher accuracy, but overall classification accuracy is preserved.  In addition
to directly addressing training imbalances, AlphaNet is also applicable to
re-balancing accuracies across biases arising from model structure or the
prioritization of different model parameters. That is, AlphaNet is deployable in
any application where the base classifiers cannot be changed, but balanced
performance is desirable -- thereby making it useful in contexts where ethics,
privacy, or intellectual property are concerns.
