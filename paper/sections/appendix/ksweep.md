# Analysis of nearest neighbor selection {#sec:ksweep}

{% raw %}
\graphicspath{{figures/appendix/}}
{% endraw %}

We analyzed the effect of the number of nearest neighbors $k$, and the
distance metric ($\mu$) used to select them, on the performance of
AlphaNet on the ImageNet-LT dataset with the \ac{cRT} model. We compared
two distance metrics: cosine distance ($\mu(z_1, z_2) = 1 - z_1^T z_2 /
\norm{z_1}_2 \norm{z_2}_2$), and Euclidean distance ($\mu(z_1, z_2) =
\norm{z_1 - z_2}_2$). For each distance metric, we performed 4 sets of
experiments, with $\rho$ in $\set{0.25, 0.5, 1}$, and varied $k$ from 2
to 10; all other hyper-parameters were kept the same as described in
@sec:impl_details.

The results are summarized in @fig:ksweep, which shows the per-split
accuracies against $k$ for different values of $\rho$ ($\rho=2$ is
omitted from this figure for space -- no special behavior was observed
for this case). We observe little change in performance beyond $k=5$,
  and also observe similar performance for both distance metrics.

Detailed results are shown in the following tables:

* Euclidean distance:

    * $\rho=0.25$:
      [@tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.25_top1;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.25_top5;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.25_semantic2_adjusted_top1;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.25_semantic3_adjusted_top1;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.25_semantic4_adjusted_top1;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.25_semantic5_adjusted_top1].

    * $\rho=0.5$:
      [@tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.5_top1;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.5_top5;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.5_semantic2_adjusted_top1;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.5_semantic3_adjusted_top1;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.5_semantic4_adjusted_top1;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.5_semantic5_adjusted_top1].

    * $\rho=1$:
      [@tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_1_top1;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_1_top5;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_1_semantic2_adjusted_top1;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_1_semantic3_adjusted_top1;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_1_semantic4_adjusted_top1;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_1_semantic5_adjusted_top1].

    * $\rho=2$:
      [@tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_2_top1;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_2_top5;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_2_semantic2_adjusted_top1;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_2_semantic3_adjusted_top1;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_2_semantic4_adjusted_top1;
      @tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_2_semantic5_adjusted_top1].

* Cosine distance:

    * $\rho=0.25$:
      [@tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.25_top1;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.25_top5;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.25_semantic2_adjusted_top1;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.25_semantic3_adjusted_top1;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.25_semantic4_adjusted_top1;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.25_semantic5_adjusted_top1].

    * $\rho=0.5$:
      [@tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.5_top1;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.5_top5;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.5_semantic2_adjusted_top1;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.5_semantic3_adjusted_top1;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.5_semantic4_adjusted_top1;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.5_semantic5_adjusted_top1].

    * $\rho=1$:
      [@tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_1_top1;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_1_top5;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_1_semantic2_adjusted_top1;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_1_semantic3_adjusted_top1;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_1_semantic4_adjusted_top1;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_1_semantic5_adjusted_top1].

    * $\rho=2$
      [@tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_2_top1;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_2_top5;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_2_semantic2_adjusted_top1;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_2_semantic3_adjusted_top1;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_2_semantic4_adjusted_top1;
      @tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_2_semantic5_adjusted_top1].

\clearpage

<div id="fig:ksweep">

![$\rho=0.25$](figures/appendix/euclidean_cosine_split_accs_vs_k_imagenetlt_crt_rho_025){#fig:nnsweep:025}

![$\rho=0.5$](figures/appendix/euclidean_cosine_split_accs_vs_k_imagenetlt_crt_rho_05){#fig:nnsweep:05}

![$\rho=1$](figures/appendix/euclidean_cosine_split_accs_vs_k_imagenetlt_crt_rho_1){#fig:nnsweep:1}

Per-split accuracies on ImageNet-LT with varying number of nearest
neighbors, for AlphaNet with \ac{cRT}.

</div>

\clearpage

{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_0.25/acc_top1.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_0.25/acc_top5.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_0.25/acc_nn_semantic_2_adjusted_top1.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_0.25/acc_nn_semantic_3_adjusted_top1.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_0.25/acc_nn_semantic_4_adjusted_top1.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_0.25/acc_nn_semantic_5_adjusted_top1.md %}

\clearpage

{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_0.5/acc_top1.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_0.5/acc_top5.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_0.5/acc_nn_semantic_2_adjusted_top1.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_0.5/acc_nn_semantic_3_adjusted_top1.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_0.5/acc_nn_semantic_4_adjusted_top1.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_0.5/acc_nn_semantic_5_adjusted_top1.md %}

\clearpage

{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_1/acc_top1.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_1/acc_top5.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_1/acc_nn_semantic_2_adjusted_top1.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_1/acc_nn_semantic_3_adjusted_top1.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_1/acc_nn_semantic_4_adjusted_top1.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_1/acc_nn_semantic_5_adjusted_top1.md %}

\clearpage

{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_2/acc_top1.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_2/acc_top5.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_2/acc_nn_semantic_2_adjusted_top1.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_2/acc_nn_semantic_3_adjusted_top1.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_2/acc_nn_semantic_4_adjusted_top1.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/rho_2/acc_nn_semantic_5_adjusted_top1.md %}

\clearpage

{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_0.25/acc_top1.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_0.25/acc_top5.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_0.25/acc_nn_semantic_2_adjusted_top1.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_0.25/acc_nn_semantic_3_adjusted_top1.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_0.25/acc_nn_semantic_4_adjusted_top1.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_0.25/acc_nn_semantic_5_adjusted_top1.md %}

\clearpage

{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_0.5/acc_top1.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_0.5/acc_top5.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_0.5/acc_nn_semantic_2_adjusted_top1.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_0.5/acc_nn_semantic_3_adjusted_top1.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_0.5/acc_nn_semantic_4_adjusted_top1.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_0.5/acc_nn_semantic_5_adjusted_top1.md %}

\clearpage

{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_1/acc_top1.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_1/acc_top5.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_1/acc_nn_semantic_2_adjusted_top1.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_1/acc_nn_semantic_3_adjusted_top1.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_1/acc_nn_semantic_4_adjusted_top1.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_1/acc_nn_semantic_5_adjusted_top1.md %}

\clearpage

{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_2/acc_top1.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_2/acc_top5.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_2/acc_nn_semantic_2_adjusted_top1.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_2/acc_nn_semantic_3_adjusted_top1.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_2/acc_nn_semantic_4_adjusted_top1.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/rho_2/acc_nn_semantic_5_adjusted_top1.md %}
