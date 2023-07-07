# Analysis of training data sampling {#sec:rhosweep}

This section contains results for AlphaNet training (following the
procedure described in @sec:impl_details) with a range of $\rho$ values.
In addition to the models described in @sec:exp, results are shown for
the \ac{LTR} model[@2022.Kong.Alshammari].

We also include results for the iNaturalist dataset using the \ac{cRT}
baseline. For this case, we trained AlphaNet for 10 epochs with a batch
size of 256. We also used smaller values of $\rho$ given the much
smaller differences in per-split accuracy. All other hyper-parameters
were kept the same as for the other datasets.

Performance was evaluated using the following metrics:

* Top-1 accuracy, i.e., if the prediction with the largest score matches the
  target label.

* Top-5 accuracy, i.e., if one of the predictions with the five largest scores
  is the target label.

* Accuracy computed by considering predictions to a nearest neighbor as
  correct, i.e., if the predicted class for a sample is one of the 5
  nearest neighbors of the true class, it is considered correct.

<!-- cSpell:ignore synsets -->

* Accuracy computed by considering predictions to a WordNet nearest
  neighbor as correct. Given a level $l$, if the predicted class for a
  sample is within $l$ nodes of the true class in the WordNet hierarchy
  (using the shortest path), it is considered correct. This analysis is
  only performed for ImageNet-LT, where image labels correspond to
  WordNet 'synsets'. We show results for $l$ in $\set{2, 3, 4, 5}$:

Results are shown in the following tables.

* ImageNet-LT:

    * \ac{cRT} baseline:
      [@tbl:main_imagenetlt_resnext50_crt_top1;
      @tbl:main_imagenetlt_resnext50_crt_top5;
      @tbl:main_imagenetlt_resnext50_crt_adjusted_top1;
      @tbl:main_imagenetlt_resnext50_crt_semantic2_adjusted_top1;
      @tbl:main_imagenetlt_resnext50_crt_semantic3_adjusted_top1;
      @tbl:main_imagenetlt_resnext50_crt_semantic4_adjusted_top1;
      @tbl:main_imagenetlt_resnext50_crt_semantic5_adjusted_top1].

    * \ac{LWS} baseline:
      [@tbl:main_imagenetlt_resnext50_lws_top1;
      @tbl:main_imagenetlt_resnext50_lws_top5;
      @tbl:main_imagenetlt_resnext50_lws_adjusted_top1;
      @tbl:main_imagenetlt_resnext50_lws_semantic2_adjusted_top1;
      @tbl:main_imagenetlt_resnext50_lws_semantic3_adjusted_top1;
      @tbl:main_imagenetlt_resnext50_lws_semantic4_adjusted_top1;
      @tbl:main_imagenetlt_resnext50_lws_semantic5_adjusted_top1].

    * \ac{RIDE} baseline:
      [@tbl:main_imagenetlt_resnext50_ride_top1;
      @tbl:main_imagenetlt_resnext50_ride_top5;
      @tbl:main_imagenetlt_resnext50_ride_adjusted_top1;
      @tbl:main_imagenetlt_resnext50_ride_semantic2_adjusted_top1;
      @tbl:main_imagenetlt_resnext50_ride_semantic3_adjusted_top1;
      @tbl:main_imagenetlt_resnext50_ride_semantic4_adjusted_top1;
      @tbl:main_imagenetlt_resnext50_ride_semantic5_adjusted_top1].

* Places-LT:

    * \ac{cRT} baseline:
      [@tbl:main_placeslt_resnet152_crt_top1;
      @tbl:main_placeslt_resnet152_crt_top5;
      @tbl:main_placeslt_resnet152_crt_adjusted_top1].

    * \ac{LWS} baseline:
      [@tbl:main_placeslt_resnet152_lws_top1;
      @tbl:main_placeslt_resnet152_lws_top5;
      @tbl:main_placeslt_resnet152_lws_adjusted_top1].

* CIFAR-100-LT:

    * \ac{RIDE} baseline:
      [@tbl:main_cifar100_resnet32_ride_top1;
      @tbl:main_cifar100_resnet32_ride_top5;
      @tbl:main_cifar100_resnet32_ride_adjusted_top1].

    * \ac{LTR} baseline:
      [@tbl:main_cifar100_resnet32_ltr_top1;
      @tbl:main_cifar100_resnet32_ltr_top5;
      @tbl:main_cifar100_resnet32_ltr_adjusted_top1].

* iNaturalist with \ac{cRT} baseline:
  [@tbl:main_inatlt_resnet152_crt_top1;
  @tbl:main_inatlt_resnet152_crt_top5;
  @tbl:main_inatlt_resnet152_crt_adjusted_top1].

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

{% include tables/appendix/models_split_top1_accs_vs_rho_inatlt.md %}
{% include tables/appendix/models_split_top5_accs_vs_rho_inatlt.md %}
