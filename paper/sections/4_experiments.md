<!-- cSpell:ignore inaturalist -->

# Experiments {#sec:exp}

## Experimental setup {#sec:exp:setup}

A detailed description of the experimental methods is contained in
@sec:impl. A short summary is presented here.

**Datasets.** We evaluated AlphaNet using three long-tailed
datasets:[^note:inaturalist] ImageNet‑LT and Places‑LT, curated by Liu
et\ al.[@2019.Yu.Liu] and CIFAR‑100‑LT, created using the procedure
described by Cui et\ al.[@2019.Belongie.Cui]. These datasets are sampled
from their respective original datasets --
ImageNet[@2015.Fei-Fei.Russakovsky], Places365[@2018.Torralba.Zhou], and
CIFAR‑100[@2009.Krizhevsky] -- such that the number of per-class
training samples has a long-tailed distribution.

The datasets are broken down into three broad splits based on the
number of training samples per class: (1) 'many' contains classes with
greater than 100 samples; (2) 'medium' contains classes with greater
than or equal to 20 samples but less than or equal to 100 samples; and
(3) 'few' contains classes with fewer than 20 samples. _The test set is
always balanced_, containing an equal number of samples for each class.
We refer to the combined 'many' and 'medium' splits as the 'base' split.

**Training data sampling.** In order to prevent over-fitting on the
'few' split samples, we used a class balanced sampling approach, using
all 'few' split samples, and a portion of the 'base' split samples.
Given $F$ 'few' split samples and a ratio $\rho$, $\rho F$ samples were
drawn from the 'base' split every epoch, with sample weights inversely
proportional to the size of their class. This ensured that all 'base'
classes had an equal probability of being
sampled.[^note:ex_sample_weights] As we show in the following section,
$\rho$ allows us to control the balance between 'few' and 'base' split
accuracy.

**Training.** All experiments used an AlphaNet with three 32 unit
layers. Unless stated otherwise, Euclidean distance was used to find
$k=5$ nearest neighbors for each 'few' split class. In this section, we
show results for $\rho$ in $\set{0.5, 1, 1.5}$. Results for a larger set
of $\rho$s are shown in @sec:rhosweep. All experiments were repeated 10
times, and we report average results.

## Long-tail classification results {#sec:exp:classres}

{% include tables/datasets_baselines_split_accs_vs_rho.md %}

**Baseline models.** First, we applied AlphaNet to models fine-tuned
using classifier re-training (cRT), and learnable weight scaling
(LWS)[@2020.Kalantidis.Kang]. These models have good overall accuracy,
but accuracy for 'few' split classes is much lower. On ImageNet‑LT,
average 'few' split accuracy using a ResNeXt‑50 backbone is around 20
points below the overall accuracy for both cRT and LWS, as seen in
@tbl:datasets_baselines_split_accs_vs_rho, which also shows other
baseline methods -- nearest classifier mean (NCM), which predicts the
nearest neighbor using average class representation, and
$\tau$‑normalized, which scales classifier weights by their
$\tau$-norm[@2020.Kalantidis.Kang].

Using features extracted from the cRT and LWS models, we used AlphaNet
to update 'few' split classifiers creating $\alpha$‑cRT and $\alpha$‑LWS
respectively. Per-split accuracies, obtained by training with $\rho$ in
$\set{0.5, 1, 1.5}$, are shown in
@tbl:datasets_baselines_split_accs_vs_rho. We get a significant increase
in the 'few' split accuracy for all values of $\rho$. Moreover, we see
that $\rho$ allows us to control the balance between 'few' split and
overall accuracies. Using larger values of $\rho$ -- i.e., training with
more 'base' split samples -- allows overall accuracy to remain closer to
the original, while still affording significant gains to 'few' split
accuracy. With $\rho=1.5$, $\alpha$‑cRT boosts 'few' split accuracy by
more than 5 points, while overall accuracy is within about 1 point.
$\alpha$‑LWS achieves even larger gains, increasing 'few' split accuracy
to around 40%, while still maintaining a competitive 48% overall
accuracy.

We repeated the above experiment on Places‑LT, where we see similar
performance gains on the 'few' split
(@tbl:datasets_baselines_split_accs_vs_rho). Notably, with $\rho=1$,
$\alpha$‑LWS increases 'few' split accuracy by about 6 points, while
overall accuracy is within 1 point of the LWS model.

{% include tables/datasets_split_accs_vs_rho_ride.md %}

{% include tables/datasets_split_accs_vs_rho_ltr.md %}

**State-of-the-art models.** Next, we applied AlphaNet to two
state-of-the-art models: (1) the 6-expert ensemble RIDE
model[@2021.Yu.Wang], and (2) the weight balancing LTR
model[@2022.Kong.Alshammari]. See @sec:impl:baselines:extract for
details on feature extraction for these models.
@tbl:datasets_split_accs_vs_rho_ride shows the base results for RIDE,
along with AlphaNet results for $\rho \in \set{0.5, 1, 1.5}$. On
ImageNet‑LT, 'few' split accuracy was increased by up to 7 points, and
on CIFAR‑100‑LT, by 5 points. For the LTR model, we show results on
CIFAR‑100‑LT in @tbl:datasets_split_accs_vs_rho_ltr -- we are able to
increase 'few' split accuracy by almost 7 points.

These results show that AlphaNet can be applied reliably with
state-of-the-art models to significantly improve the accuracy for rare
classes.

## Comparison with control {#sec:exp:control}

Our method is based on the core hypothesis that classifiers can be
improved using nearest neighbors. In this section, we directly evaluate
this hypothesis. Based on the results in the previous section, the
improvements in 'few' split accuracy could be attributed simply to the
extra fine-tuning of the classifiers. So, using the cRT model on
ImageNet‑LT, we retrained AlphaNet with 5 randomly chosen 'base' split
classes as "neighbors" for each 'few' split class. This differs from
our previous experiments only in the classes used to update 'few' split
classifiers, so if AlphaNet's improvements were solely due to extra
fine-tuning, we should see similar results. However, as seen in
@fig:euclidean_random_split_deltas_vs_imagenetlt_crt, training with
nearest neighbors selected by Euclidean distance garners much larger
improvements in 'few' split accuracy, with similar trends in overall
accuracy. This supports our hypothesis that classifiers for data-poor
classes can make use of information from visually similar classes to
improve classification performance.

## Prediction changes {#sec:exp:predchanges}

As shown in @sec:intro, the cRT model frequently misclassifies 'few'
split classes as visually similar 'base' split classes. Using the
AlphaNet model with $\rho=0.5$, we performed the same analyses as
before. @fig:pred_changes:few shows the change in sample predictions,
where we see that a large portion of samples previously misclassified as
a nearest neighbor are correctly classified after their classifiers are
updated with AlphaNet. Furthermore, as seen in @fig:cls_delta_vs_nndist,
AlphaNet improvements are strongly correlated to mean nearest neighbor
distance. Classes with close neighbors, which had a high likelihood of
being misclassified by the baseline model, see the biggest improvement
in test accuracy.

![Change in per-class test accuracy for 'few' split of ImageNet‑LT with
$\alpha$‑cRT, versus mean Euclidean distance to 5 nearest neighbors.
Comparing with @fig:analysis:acc_vs_dist, we see that AlphaNet provides
the largest boost to classes with close nearest neighbors, which have
poor baseline
performance.](figures/cls_delta_vs_nndist_imagenetlt_crt_rho_05){#fig:cls_delta_vs_nndist}

![Change in split accuracies for $\alpha$‑cRT on ImageNet‑LT. For each
value of $\rho$, the two plots show the raw difference in split accuracy
(with accuracy expressed as a fraction) for AlphaNet compared to the
baseline cRT model. Left shows the results for normal training with 5
nearest neighbors by Euclidean distance, and right shows the results for
training with 5 random "neighbors" for each 'few' split class. Training
with nearest neighbors leads to a larger increase in 'few' split
accuracy, especially for small $\rho$, which cannot be accounted for by
the additional fine-tuning of classifiers
alone.](figures/euclidean_random_split_deltas_vs_rho_imagenetlt_crt){#fig:euclidean_random_split_deltas_vs_imagenetlt_crt}

<div id="fig:pred_changes">

![Predictions on 'few' split classes, with NNs selected from the 'base'
split.](figures/few_pred_changes_nn_base_imagenetlt_crt_rho_05){#fig:pred_changes:few}

![Predictions on 'base' split classes, with NNs selected from the 'few'
split.](figures/base_pred_changes_nn_few_imagenetlt_crt_rho_05){#fig:pred_changes:base}

![All predictions, with NNs selected from all classes. The hatched
portions represent the 'few'
split.](figures/all_pred_changes_nn_all_imagenetlt_crt_rho_05){#fig:pred_changes:all}

Change in sample predictions for $\alpha$‑cRT ($\rho=0.5$) on
ImageNet‑LT. For each plot, the bars on the left show the distribution
of predictions by the baseline model; and the bars on the right show the
distribution for $\alpha$‑cRT. The groupings follow the scheme described
in @fig:analysis:bins. The counts are aggregated from 10 repetitions of
training $\alpha$‑cRT. The "flow" bands from left to right show the
changes in individual sample predictions.

</div>

<div id="fig:pred_changes_semantic">

![Predictions on 'few' split
classes.](figures/few_pred_changes_nn_semantic4_imagenetlt_crt_rho_05){#fig:pred_changes_semantic:few}

![Predictions on 'base' split
classes.](figures/base_pred_changes_nn_semantic4_imagenetlt_crt_rho_05){#fig:pred_changes_semantic:base}

![All predictions; hatched portions represent 'few'
split.](figures/all_pred_changes_nn_semantic4_imagenetlt_crt_rho_05){#fig:pred_changes_semantic:all}

Change in sample predictions for $\alpha$‑cRT ($\rho=0.5$) grouped with
respect to nearest neighbors identified using WordNet. This figure shows
the same results as @fig:pred_changes, but grouped using differently
defined nearest neighbors -- the new nearest neighbors are only used for
visualizing.

</div>

## Analysis of AlphaNet predictions {#sec:exp:analysis}

AlphaNet significantly boosts the accuracy of 'few' split classes.
However, we do see a decrease in overall accuracy compared to baseline
models, particularly for small values of $\rho$. It is important to note
that the increase in 'few' split accuracy is much larger than the
decrease in overall accuracy. As discussed earlier, in many applications
it is important to have balanced performance across classes, and
AlphaNet succeeds in making accuracies more balanced across splits.

However, we further analyzed the prediction changes for 'base' split
samples. Specifically, @fig:pred_changes:base shows change in
predictions for 'base' split samples, with nearest neighbors selected
from the 'few' split. We see a small increase in misclassifications as
'few' split classes. This leads to the slight decrease in overall
accuracy, which is also evident in @fig:pred_changes:all where all
predictions are shown, and with nearest neighbors from all classes.

This previous analysis was conducted using nearest neighbors identified
based on visual similarity. Since this is dependent on the
representation space of the particular model, we conducted an additional
analysis to see the behavior of predictions with respect to
_semantically similar_ categories. For classes in ImageNet‑LT, we
defined nearest neighbors using distance in the WordNet[@2010.Princeton]
hierarchy. Specifically, if two classes (e.g., 'Lhasa' and 'Tibetan
terrier') share a parent at most 4 levels higher in WordNet (in this
example, 'dog'), we consider them to be one of each other's nearest
neighbors. @fig:pred_changes_semantic shows $\alpha$‑cRT predictions
grouped using nearest neighbors defined his way. We see that a
large number of incorrect predictions are to semantically similar
categories which can be hard for even humans to distinguish. This
suggests that metrics for long-tail classification should be
re-evaluated for large datasets with many similar classes. Considering
only misclassifications to semantically dissimilar classes, we see that
AlphaNet still improves performance on 'few' split classes, while
maintaining overall accuracy. So, despite model-specific visual
similarity, AlphaNet garners improvements at the semantic level, showing
that it can be applied to models beyond those used in this paper.

[^note:inaturalist]: Another popular dataset for evaluating long-tail
    models is iNaturalist[@2018.Belongie.Horn]. However, models are able
    to achieve much more balanced results on this dataset, compared to
    other long-tailed datasets. For example, with the cRT model, 'few'
    split accuracy (69.2%) is only 2 points lower than the overall
    accuracy (71.2%). So the dataset does not represent a valid use case
    for our proposed method, and we omitted the dataset from our main
    experiments. Results for this dataset are included in the appendix
    (@sec:rhosweep).

[^note:ex_sample_weights]: For example, suppose there are 2 'base'
    classes -- class\ 1 has 10 samples, and class\ 2 has 100 samples.
    Then, each class\ 1 sample is assigned a weight of 0.1, and each
    class\ 2 sample is assigned a weight of 0.01. Sampling with this
    weight distribution, both classes have a 50% chance of being
    sampled.
