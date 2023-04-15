# Additional Results

This section contains additional analysis, as well as results on the
\ac{LTR} model\ [@2022.Kong.Alshammari]. The included results are

* Per-split accuracy with varying number of nearest numbers computed
  using Euclidean and cosine distance
  (@fig:euclidean_cosine_split_accs_vs_k_imagenetlt_crt).

* Distribution of $\alpha$ values across 'few' split classes for
  different number of nearest numbers computed using Euclidean
  distance (@fig:ks_alpha_dists_imagenetlt_crt).

* Tables of per-split accuracy comparing AlphaNet trained on a range
  of $\rho$ values:

    * ImageNet-LT ([@tbl:imgnetlt_crt_full;@tbl:imgnetlt_lws_full;@tbl:imgnetlt_ride_full]).

    * Places-LT ([@tbl:placeslt_crt_full;@tbl:placeslt_lws_full]).

    * CIFAR-100-LT ([@tbl:cifarlt_ride_full;@tbl:cifarlt_ltr_full]).

* Change in per-class accuracy for AlphaNet compared to baseline models:

    * ImageNet-LT ([@fig:cls_deltas_imagenetlt_crt;@fig:cls_deltas_imagenetlt_lws;@fig:cls_deltas_imagenetlt_ride]).

    * Places-LT ([@fig:cls_deltas_placeslt_crt;@fig:cls_deltas_placeslt_lws]).

    * CIFAR-100-LT ([@fig:cls_deltas_cifar100_ride;@fig:cls_deltas_cifar100_ltr]).

* Per-class test accuracy vs. mean distance to 5 nearest neighbors:

    * ImageNet-LT ([@fig:rhos_cls_delta_vs_nndist_imagenetlt_crt;@fig:rhos_cls_delta_vs_nndist_imagenetlt_lws;@fig:rhos_cls_delta_vs_nndist_imagenetlt_ride]).

    * Places-LT ([@fig:rhos_cls_delta_vs_nndist_placeslt_crt;@fig:rhos_cls_delta_vs_nndist_placeslt_lws]).

    * CIFAR-100-LT ([@fig:rhos_cls_delta_vs_nndist_cifar100_ride;@fig:rhos_cls_delta_vs_nndist_cifar100_ltr]).

* Distribution of $\alpha$ values across 'few' split classes:

    * ImageNet-LT ([@fig:rhos_alpha_dists_imagenetlt_crt;@fig:rhos_alpha_dists_imagenetlt_lws;@fig:rhos_alpha_dists_imagenetlt_ride]).

    * Places-LT ([@fig:rhos_alpha_dists_placeslt_crt;@fig:rhos_alpha_dists_placeslt_lws]).

    * CIFAR-100-LT ([@fig:rhos_alpha_dists_cifar100_ride;@fig:rhos_alpha_dists_cifar100_ltr]).


{% include tables/appendix/imagenetlt_crt_full.md %}

{% include tables/appendix/imagenetlt_lws_full.md %}

{% include tables/appendix/imagenetlt_ride_full.md %}


{% include tables/appendix/placeslt_crt_full.md %}

{% include tables/appendix/placeslt_lws_full.md %}


{% include tables/appendix/cifarlt_ride_full.md %}

{% include tables/appendix/cifarlt_ltr_full.md %}


\clearpage


<div id="fig:euclidean_cosine_split_accs_vs_k_imagenetlt_crt">

![$\rho=0.5$](figures/appendix/euclidean_cosine_split_accs_vs_k_imagenetlt_crt_rho_05){#fig:euclidean_cosine_split_accs_vs_k_imagenetlt_crt_rho_05}

![$\rho=1$](figures/appendix/euclidean_cosine_split_accs_vs_k_imagenetlt_crt_rho_1){#fig:euclidean_cosine_split_accs_vs_k_imagenetlt_crt_rho_1}

![$\rho=2$](figures/appendix/euclidean_cosine_split_accs_vs_k_imagenetlt_crt_rho_2){#fig:euclidean_cosine_split_accs_vs_k_imagenetlt_crt_rho_2}

Per-split accuracies on ImageNet-LT with varying number of nearest neighbors for AlphaNet on \ac{cRT} baseline.

</div>


<div id="fig:ks_alpha_dists_imagenetlt_crt">

![$\rho=0.5$](figures/appendix/ks_alpha_dists_imagenetlt_crt_rho_05){#fig:ks_alpha_dists_imagenetlt_crt_rho_05}

![$\rho=1$](figures/appendix/ks_alpha_dists_imagenetlt_crt_rho_1){#fig:ks_alpha_dists_imagenetlt_crt_rho_1}

![$\rho=2$](figures/appendix/ks_alpha_dists_imagenetlt_crt_rho_2){#fig:ks_alpha_dists_imagenetlt_crt_rho_2}

Distribution of $\alpha$ values across $k$ for AlphaNet with \ac{cRT} baseline trained on ImageNet-LT.

</div>


![Change in per-class accuracy on ImageNet-LT with AlphaNet on \ac{cRT}. The $\Delta$s indicate the average change in split accuracy.](figures/appendix/cls_deltas_imagenetlt_crt){#fig:cls_deltas_imagenetlt_crt}

![Change in per-class accuracy on ImageNet-LT with AlphaNet on \ac{LWS}. The $\Delta$s indicate the average change in split accuracy.](figures/appendix/cls_deltas_imagenetlt_lws){#fig:cls_deltas_imagenetlt_lws}

![Change in per-class accuracy on ImageNet-LT with AlphaNet on \ac{RIDE}. The $\Delta$s indicate the average change in split accuracy.](figures/appendix/cls_deltas_imagenetlt_ride){#fig:cls_deltas_imagenetlt_ride}


![Change in per-class accuracy on Places-LT with AlphaNet on \ac{cRT}. The $\Delta$s indicate the average change in split accuracy.](figures/appendix/cls_deltas_placeslt_crt){#fig:cls_deltas_placeslt_crt}

![Change in per-class accuracy on Places-LT with AlphaNet on \ac{LWS}. The $\Delta$s indicate the average change in split accuracy.](figures/appendix/cls_deltas_placeslt_lws){#fig:cls_deltas_placeslt_lws}


![Change in per-class accuracy on CIFAR-100-LT with AlphaNet on \ac{RIDE}. The $\Delta$s indicate the average change in split accuracy.](figures/appendix/cls_deltas_cifar100_ride){#fig:cls_deltas_cifar100_ride}

![Change in per-class accuracy on CIFAR-100-LT with AlphaNet on \ac{LTR}. The $\Delta$s indicate the average change in split accuracy.](figures/appendix/cls_deltas_cifar100_ltr){#fig:cls_deltas_cifar100_ltr}


<div id="fig:rhos_cls_delta_vs_nndist_imagenetlt">

![\ac{cRT}](figures/appendix/rhos_cls_delta_vs_nndist_imagenetlt_crt){#fig:rhos_cls_delta_vs_nndist_imagenetlt_crt}

![\ac{LWS}](figures/appendix/rhos_cls_delta_vs_nndist_imagenetlt_lws){#fig:rhos_cls_delta_vs_nndist_imagenetlt_lws}

![\ac{RIDE}](figures/appendix/rhos_cls_delta_vs_nndist_imagenetlt_ride){#fig:rhos_cls_delta_vs_nndist_imagenetlt_ride}

Test accuracy on ImageNet-LT vs. mean nearest neighbor distance for AlphaNet trained with $k=5$ neighbors. The lines show regression fits, and the $r$ values in the upper right are Pearson correlations.

</div>


<div id="fig:rhos_cls_delta_vs_nndist_placeslt">

![\ac{cRT}](figures/appendix/rhos_cls_delta_vs_nndist_placeslt_crt){#fig:rhos_cls_delta_vs_nndist_placeslt_crt}

![\ac{LWS}](figures/appendix/rhos_cls_delta_vs_nndist_placeslt_lws){#fig:rhos_cls_delta_vs_nndist_placeslt_lws}

Test accuracy on Places-LT vs. mean nearest neighbor distance for AlphaNet trained with $k=5$ neighbors. The lines show regression fits, and the $r$ values in the upper right are Pearson correlations.

</div>


<div id="fig:rhos_cls_delta_vs_nndist_cifar100">

![\ac{RIDE}](figures/appendix/rhos_cls_delta_vs_nndist_cifar100_ride){#fig:rhos_cls_delta_vs_nndist_cifar100_ride}

![\ac{LTR}](figures/appendix/rhos_cls_delta_vs_nndist_cifar100_ltr){#fig:rhos_cls_delta_vs_nndist_cifar100_ltr}

Test accuracy on CIFAR-100-LT vs. mean nearest neighbor distance for AlphaNet trained with $k=5$ neighbors. The lines show regression fits, and the $r$ values in the upper right are Pearson correlations.

</div>


<div id="fig:rhos_alpha_dists_imagenetlt">

![\ac{cRT}](figures/appendix/rhos_alpha_dists_imagenetlt_crt){#fig:rhos_alpha_dists_imagenetlt_crt}

![\ac{LWS}](figures/appendix/rhos_alpha_dists_imagenetlt_lws){#fig:rhos_alpha_dists_imagenetlt_lws}

![\ac{RIDE}](figures/appendix/rhos_alpha_dists_imagenetlt_ride){#fig:rhos_alpha_dists_imagenetlt_ride}

Distribution of $\alpha$s for AlphaNet trained on ImageNet-LT.

</div>


<div id="fig:rhos_alpha_dists_placeslt">

![\ac{cRT}](figures/appendix/rhos_alpha_dists_placeslt_crt){#fig:rhos_alpha_dists_placeslt_crt}

![\ac{LWS}](figures/appendix/rhos_alpha_dists_placeslt_lws){#fig:rhos_alpha_dists_placeslt_lws}

Distribution of $\alpha$s for AlphaNet trained on Places-LT.

</div>


<div id="fig:rhos_alpha_dists_cifar100">

![\ac{RIDE}](figures/appendix/rhos_alpha_dists_cifar100_ride){#fig:rhos_alpha_dists_cifar100_ride}

![\ac{LTR}](figures/appendix/rhos_alpha_dists_cifar100_ltr){#fig:rhos_alpha_dists_cifar100_ltr}

Distribution of $\alpha$s for AlphaNet trained on CIFAR-100-LT.

</div>
