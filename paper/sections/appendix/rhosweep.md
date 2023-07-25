<!-- cSpell:ignore synsets -->

# Analysis of training data sampling {#sec:rhosweep}

This section contains results for AlphaNet training with a range of
$\rho$ values. Training was performed following the same procedure as
described in @sec:impl. We also include results for the iNaturalist
dataset using the cRT baseline. For iNaturalist, we used smaller values
of $\rho$ given the much smaller differences in per-split accuracy.

The results are summarized in @fig:models_split_top1_deltas_vs_rho and
@fig:models_split_top5_deltas_vs_rho, which show change in per-split
top‑1 and top‑5 accuracy (respectively), versus $\rho$ (iNaturalist
results are omitted from these figures due to the different set of
$\rho$s used).

Detailed results, organized by dataset, are shown in the following
tables:

* ImageNet‑LT:
    * Top‑1 accuracy: @tbl:models_split_top1_accs_vs_rho_imagenetlt.
    * Top‑5 accuracy: @tbl:models_split_top5_accs_vs_rho_imagenetlt.

* Places‑LT:
    * Top‑1 accuracy: @tbl:models_split_top1_accs_vs_rho_placeslt.
    * Top‑5 accuracy: @tbl:models_split_top5_accs_vs_rho_placeslt.

* CIFAR‑100‑LT:
    * Top‑1 accuracy: @tbl:models_split_top1_accs_vs_rho_cifarlt.
    * Top‑5 accuracy: @tbl:models_split_top5_accs_vs_rho_cifarlt.

* iNaturalist:
    * Top‑1 accuracy: @tbl:models_split_top1_accs_vs_rho_inat.
    * Top‑5 accuracy: @tbl:models_split_top5_accs_vs_rho_inat.

In addition to top‑1 and top‑5 accuracy, we evaluated performance on
ImageNet‑LT by considering predictions to a WordNet[@2010.Princeton]
nearest neighbor as correct.[^note:semantic_acc] Given a level $l$, if
the predicted class for a sample is within $l$ nodes of the true class
in the WordNet hierarchy (using the shortest path), it is considered
correct. We used $l=4$, and these results are shown in
@tbl:models_split_semantic4_accs_vs_rho_imagenetlt.

[^note:semantic_acc]: This is only possible for ImageNet‑LT since image
    labels correspond to WordNet synsets.

\clearpage

![Change in per-split top‑1 accuracy vs. $\rho$ for alphanet training
with different baseline models, and on different
datasets](figures/appendix/models_split_top1_deltas_vs_rho){#fig:models_split_top1_deltas_vs_rho}

\clearpage

![Change in per-split top‑5 accuracy vs. $\rho$ for alphanet training
with different baseline models, and on different
datasets](figures/appendix/models_split_top5_deltas_vs_rho){#fig:models_split_top5_deltas_vs_rho}

\clearpage

{% include tables/appendix/models_split_top1_accs_vs_rho_imagenetlt.md %}
{% include tables/appendix/models_split_top5_accs_vs_rho_imagenetlt.md %}
{% include tables/appendix/models_split_semantic4_accs_vs_rho_imagenetlt.md %}

\clearpage

{% include tables/appendix/models_split_top1_accs_vs_rho_placeslt.md %}
{% include tables/appendix/models_split_top5_accs_vs_rho_placeslt.md %}

\clearpage

{% include tables/appendix/models_split_top1_accs_vs_rho_cifarlt.md %}
{% include tables/appendix/models_split_top5_accs_vs_rho_cifarlt.md %}

\clearpage

{% include tables/appendix/models_split_top1_accs_vs_rho_inat.md %}
{% include tables/appendix/models_split_top5_accs_vs_rho_inat.md %}

\clearpage
