{% raw %}
\graphicspath{{figures/appendix/}}
{% endraw %}

# Analysis of nearest neighbor selection {#sec:ksweep}

We analyzed the effect of the number of nearest neighbors $k$, and the
distance metric ($\mu$), on the performance of AlphaNet, using the cRT
model on ImageNet‑LT. We compared two distance metrics:

* Cosine distance: $\mu(z_1, z_2) = 1 - z_1^T z_2$.
* Euclidean distance: $\mu(z_1, z_2) = \norm{z_1 - z_2}_2$.

For each distance metric, we performed 4 sets of experiments, with
$\rho$ in $\set{0.25, 0.5, 1, 2}$. For each $\rho$, we varied $k$ from 2
to 10; all other hyper-parameters were kept the same as described in
@sec:impl.

The results are summarized in @fig:ksweep, which shows per-split top‑1
accuracies against $k$ for different values of $\rho$ ($\rho=2$ is
omitted from this figure for space -- no special behavior was observed
for this case). We observe little change in performance beyond $k=5$,
and also observe similar performance for both distance metrics.

The full set of top‑1 and top‑5 accuracies is shown in the following
tables:

* Euclidean distance:
    * Top‑1 accuracy:
      @tbl:rhos_split_top1_accs_vs_k_imagenetlt_crt_euclidean.
    * Top‑5 accuracy:
      @tbl:rhos_split_top5_accs_vs_k_imagenetlt_crt_euclidean.

* Cosine distance:
    * Top‑1 accuracy:
      @tbl:rhos_split_top1_accs_vs_k_imagenetlt_crt_cosine.
    * Top‑5 accuracy:
      @tbl:rhos_split_top5_accs_vs_k_imagenetlt_crt_cosine.

\clearpage

<div id="fig:ksweep">

![$\rho=0.25$](figures/appendix/euclidean_cosine_split_accs_vs_k_imagenetlt_crt_rho_025){#fig:ksweep:025}

![$\rho=0.5$](figures/appendix/euclidean_cosine_split_accs_vs_k_imagenetlt_crt_rho_05){#fig:ksweep:05}

![$\rho=1$](figures/appendix/euclidean_cosine_split_accs_vs_k_imagenetlt_crt_rho_1){#fig:ksweep:1}

Per-split test accuracies for $\alpha$‑cRT on ImageNet‑LT versus the
number of nearest neighbors $k$.

</div>

\clearpage

{% include tables/appendix/rhos_split_top1_accs_vs_k_imagenetlt_crt_euclidean.md %}
{% include tables/appendix/rhos_split_top5_accs_vs_k_imagenetlt_crt_euclidean.md %}

\clearpage

{% include tables/appendix/rhos_split_top1_accs_vs_k_imagenetlt_crt_cosine.md %}
{% include tables/appendix/rhos_split_top5_accs_vs_k_imagenetlt_crt_cosine.md %}

\clearpage
