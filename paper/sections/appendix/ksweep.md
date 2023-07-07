# Analysis of nearest neighbor selection {#sec:ksweep}

{% raw %}
\graphicspath{{figures/appendix/}}
{% endraw %}

We analyzed the effect of the number of nearest neighbors $k$, and the
distance metric ($\mu$), on the performance of AlphaNet on the
ImageNet-LT dataset with the \ac{cRT} model. We compared two distance
metrics: cosine distance ($\mu(z_1, z_2) = 1 - z_1^T z_2 / \norm{z_1}_2
\norm{z_2}_2$), and Euclidean distance ($\mu(z_1, z_2) = \norm{z_1 -
z_2}_2$). For each distance metric, we performed 4 sets of experiments,
with $\rho$ in $\set{0.25, 0.5, 1, 2}$. For each $\rho$, we varied $k$
from 2 to 10; all other hyper-parameters were kept the same as described
in @sec:impl_details.

The results are summarized in @fig:ksweep, which shows per-split top-1
accuracies against $k$ for different values of $\rho$ ($\rho=2$ is
omitted from this figure for space -- no special behavior was observed
for this case). We observe little change in performance beyond $k=5$,
and also observe similar performance for both distance metrics.

The full set of top-1 and top-5 accuracies is shown in
[@tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_top1;
@tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_top5;
@tbl:nnsweep_cosine_imagenetlt_resnext50_crt_top1;
@tbl:nnsweep_cosine_imagenetlt_resnext50_crt_top5].

\clearpage

<div id="fig:ksweep">

![$\rho=0.25$](figures/appendix/euclidean_cosine_split_accs_vs_k_imagenetlt_crt_rho_025){#fig:nnsweep:025}

![$\rho=0.5$](figures/appendix/euclidean_cosine_split_accs_vs_k_imagenetlt_crt_rho_05){#fig:nnsweep:05}

![$\rho=1$](figures/appendix/euclidean_cosine_split_accs_vs_k_imagenetlt_crt_rho_1){#fig:nnsweep:1}

Per-split accuracies on ImageNet-LT with varying number of nearest
neighbors, for AlphaNet with \ac{cRT}.

</div>

\clearpage

{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/acc_top1.md %}
{% include tables/appendix/nnsweep_euclidean/imagenetlt_resnext50_crt/acc_top5.md %}

\clearpage

{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/acc_top1.md %}
{% include tables/appendix/nnsweep_cosine/imagenetlt_resnext50_crt/acc_top5.md %}