<!-- cSpell:ignore inaturalist -->

# Experiments {#sec:exp}

## Experimental setup {#sec:exp:setup}

A detailed description of the experimental methods is contained in
@sec:impl_details. A short summary is presented in this section.

**Datasets.** We evaluated AlphaNet using three long-tailed
datasets:[^note:inaturalist] ImageNet‑LT, Places‑LT,[@2019.Yu.Liu] and
CIFAR‑100‑LT.[@2022.Kong.Alshammari] These datasets are sampled from
their respective original datasets, ImageNet,[@2015.Fei-Fei.Russakovsky]
Places365,[@2017.Torralba.Zhou] and CIFAR‑100,[@2009.Hinton.Krizhevsky]
such that the new distributions follow a standard long-tailed
distribution.

The datasets are broken down into three broad splits that indicate the
number of training samples per class: (1) 'many' contains classes with
greater than 100 samples; (2) 'medium' contains classes with greater
than or equal to 20 samples but less than or equal to 100 samples; and
(3) 'few' contains classes with fewer than 20 samples. **The test set is
always balanced**, containing an equal number of samples for each class.
We refer to the combined 'many' and 'medium' splits as the 'base' split.

**Training data sampling.** In order to prevent over-fitting on the
'few' split samples, we used a class balanced sampling approach, using
all 'few' split samples, and a portion of the 'base' split samples.
Given $F$ 'few' split samples, and a ratio $\rho$, every epoch, $\rho F$
samples were drawn from the 'base' split, with sample weights inversely
proportional to size of their class. This ensured that all 'base'
classes had an equal probability of being
sampled.[^note:ex_sample_weights] As we show in the following section,
$\rho$ allows us to control the balance between 'few' and 'base' split
accuracy.

**Training.** All experiments used an AlphaNet module with three 32 unit
layers. Unless stated otherwise, Euclidean distance was used to find
$k=5$ nearest neighbors for each 'few' split class. All experiments were
repeated 10 times, and we report average results.

## Long-tail classification results {#sec:exp:classres}

{% include tables/datasets_baselines_split_accs_vs_rho.md %}

**Baseline models.** First, we applied AlphaNet on models learned using
classifier re-training (cRT) and learnable weight scaling
(LWS).[@2019.Kalantidis.Kang] These methods have good overall accuracy,
but accuracy for 'few' split classes is much lower. On the ImageNet‑LT
dataset, average 'few' split accuracy using a ResNeXt‑50 backbone is
nearly 20 points below the overall accuracy, as seen in
@tbl:datasets_baselines_split_accs_vs_rho, which also shows other
baseline methods -- nearest classifier mean (NCM), which predicts the
nearest neighbor using average class representation, and
$\tau$‑normalized, which re-balances classifiers by adjusting their
classifier weights.[@2019.Kalantidis.Kang] Using features extracted from
cRT and LWS models, we used AlphaNet to update 'few' split classifiers.
On both models, we saw a significant increase in the 'few' split
accuracy for all values of $\rho$. For $\rho = 1$, average 'few' split
accuracy was boosted by 7 points for the cRT model, and about 11 points
for the LWS model.

We repeated the above experiment on the Places‑LT dataset, where again
'few' split accuracy for the cRT and LWS models is much lower than the
overall accuracy (by around 12 and 9 points respectively), as seen in
@tbl:datasets_baselines_split_accs_vs_rho. With $\rho = 1$, AlphaNet
improved 'few' split accuracy by 2 points on average for the cRT model,
and about 6 points on average for the LWS model.

{% include tables/datasets_split_accs_vs_rho_ride.md %}

{% include tables/datasets_split_accs_vs_rho_ltr.md %}

**State-of-the-art models.** Next, we applied AlphaNet on two state-of-the-art
models: (1) the 6-expert ensemble RIDE model,[@2020.Yu.Wang] and (2) the
weight balancing LTR model.[@2022.Kong.Alshammari]
@tbl:datasets_split_accs_vs_rho_ride shows the base results for RIDE,
along with AlphaNet results for $\rho \in \set{0.5, 1, 2}$. On
ImageNet‑LT, 'few' split accuracy was increased by up to 7 points, and
on CIFAR‑100‑LT, by 5 points. For the LTR model, we show results on
CIFAR‑100‑LT in @tbl:datasets_split_accs_vs_rho_ltr. With $\rho=1$,
'few' split accuracy is increased by almost 9 points, while with $\rho=3$,
we maintain overall accuracy within 1 point, while still improving 'few'
split accuracy by about 4 points on average.

These results show that AlphaNet can be applied reliably with
state-of-the-art models to significantly improve the accuracy of rare
classes.

## Comparison with control {#sec:exp:control}

![Change in split accuracy[^note:acc_delta] for AlphaNet training of cRT model on
ImageNet‑LT. For each value of $\rho$, the two plots show the raw change
in split accuracy for AlphaNet compared to the baseline cRT model, as
well as the change in overall accuracy. Left shows the results for
normal training with 5 euclidean nearest neighbors, and right shows the
results for training with 5 random neighbors for each 'few' split class.
Training with nearest neighbors leads to a larger increase in 'few'
split accuracy (especially for small values of $\rho$), which cannot be
accounted for by the additional fine-tuning of classifiers
alone.](figures/euclidean_random_split_deltas_vs_rho_imagenetlt_crt){#fig:euclidean_random_split_deltas_vs_imagenetlt_crt}

Our method is based on the core hypothesis that classifiers can be
improved using nearest neighbors. In this section, we directly evaluate
this hypothesis. Based on the results in the previous section, the
improvements in 'few' split accuracy could be attributed simply to the
extra fine-tuning of the classifiers. So, using the cRT model on
ImageNet‑LT, we retrained AlphaNet using 5 randomly chosen neighbors for
each 'few' split class, rather than nearest neighbors. This differs from
our previous experiments only in the nature of neighbors used, so if our
method's improvements were solely due to extra fine-tuning, we should
see similar results. However, as seen in
@fig:euclidean_random_split_deltas_vs_imagenetlt_crt, training with
nearest neighbors garners much larger improvements in 'few' split
accuracy, with similar trends in overall accuracy. This supports our
hypothesis that data-poor classes can make use of information from
neighbors to improve classification performance.

## Prediction changes {#sec:exp:predchanges}

![Change in per-class accuracy for AlphaNet applied over cRT features,
versus mean Euclidean distance to five nearest neighbors. Comparing with
@fig:analysis:acc_vs_dist, we can see that AlphaNet provides the largest
boost to classes with poor baseline performance, which have close
nearest
neighbors.](figures/cls_delta_vs_nndist_imagenetlt_crt_rho_05){#fig:cls_delta_vs_nndist}

As shown in @sec:intro, the cRT model frequently misclassifies 'few'
split classes as visually similar 'base' split classes. Using the
AlphaNet model with $\rho=0.5$, we performed the same analyses as
before. @fig:pred_changes:few shows the change in sample predictions,
where we see that a large portion of samples previously misclassified as
a nearest neighbor are correctly classified after their classes are
updated with AlphaNet. Furthermore, as seen in @fig:cls_delta_vs_nndist,
AlphaNet improvements are strongly correlated to mean nearest neighbor
distance. Classes with close neighbors, which had a high likelihood of
being misclassified by the baseline model, see the biggest improvement
in test accuracy.

## Analysis of AlphaNet predictions {#sec:exp:analysis}

<div id="fig:pred_changes">

![Predictions on 'few' split classes, with nearest neighbors selected
from the 'base'
split.](figures/few_pred_changes_nn_base_imagenetlt_crt_rho_05){#fig:pred_changes:few}

![Predictions on 'base' split classes, with nearest neighbors selected
from the 'few'
split.](figures/base_pred_changes_nn_few_imagenetlt_crt_rho_05){#fig:pred_changes:base}

![All predictions, with nearest neighbors selected from all
classes.](figures/all_pred_changes_nn_all_imagenetlt_crt_rho_05){#fig:pred_changes:all}

Change in sample predictions for AlphaNet applied to cRT features, using
$k=5$ nearest neighbors by Euclidean distance. The bars on the left
show the distribution of predictions by the baseline model; and the bars
on the right show the distribution for AlphaNet. The counts are
aggregated from 10 independent runs of AlphaNet. The "flow" bands from
left to right show the changes in individual sample predictions.

</div>

AlphaNet significantly boosts the accuracy of 'few' split classes.
However, looking at @tbl:datasets_baselines_split_accs_vs_rho and
@tbl:datasets_split_accs_vs_rho_ride, we see that the overall accuracy
decreases compared to baseline models, particularly for small values of
$\rho$. It is important to note that the increase in 'few' split
accuracy is much larger than the decrease in overall accuracy. As
discussed earlier, in many applications it is important to have balanced
performance across classes, and AlphaNet succeeds in making accuracies
more balanced across splits.

However, we further analyzed the prediction changes for 'base' split
samples. Specifically, @fig:pred_changes:base shows change in
predictions for 'base' split samples, with nearest neighbors selected
from the 'few' split. We see a small increase in misclassifications as
'few' split classes. This leads to the slight decrease in overall
accuracy, which is also evident in @fig:pred_changes:all where all
predictions are shown, and with nearest neighbors from all classes.

<div id="fig:pred_changes_semantic">

![Predictions on 'few' split
classes.](figures/few_pred_changes_nn_semantic4_imagenetlt_crt_rho_05){#fig:pred_changes_semantic:few}

![Predictions on 'base' split
classes.](figures/base_pred_changes_nn_semantic4_imagenetlt_crt_rho_05){#fig:pred_changes_semantic:base}

![All
predictions.](figures/all_pred_changes_nn_semantic4_imagenetlt_crt_rho_05){#fig:pred_changes_semantic:all}

Change in sample predictions for AlphaNet applied for cRT features, with
nearest neighbors identified using WordNet categories. This figure
represents the same predictions as @fig:pred_changes, but grouped
differently.

</div>

The previous analysis was conducted using nearest neighbors identified
based on visual similarity. Since this is dependent on the particular
model, we conducted an additional analysis to see the behavior of
predictions with respect to _semantically similar_ categories. For
classes in ImageNet‑LT, we defined nearest neighbors using distance in
the WordNet[@wordnet] hierarchy. Specifically, if two classes
(e.g., 'Lhasa' and 'Tibetan terrier') share a parent at most 4 levels
higher in WordNet (in this example, 'dog'), we consider them to be
nearest neighbors. @fig:pred_changes_semantic shows the predictions for
AlphaNet with cRT grouped based on these nearest neighbors. As we can
see, a large number of predictions which are considered incorrect are
among semantically similar categories which can be hard for even humans
to distinguish. This suggests that metrics for long-tail classification
might need to be re-evaluated for large datasets with many similar
classes.

[^note:inaturalist]: Another popular dataset used for testing long-tail
    models is iNaturalist.[@2018.Belongie.Horn] Results for this
    dataset, however, are much more balanced across splits. For example,
    the cRT model achieves the following per-split accuracies: 75.9%
    ('many'), 71.4% ('medium'), and 70.4% ('few'). The 'few' split
    accuracy is only 0.8 points lower than the overall accuracy (71.2%);
    so the dataset does not represent a valid use case for our proposed
    method, and we omitted the dataset from our main experiments.
    Results for this dataset are included in the appendix
    (@sec:rhosweep).

[^note:ex_sample_weights]: For example, suppose there are 2 'base'
    classes -- class\ 1 has 10 samples, and class\ 2 has 100 samples.
    Then, each class\ 1 sample is assigned a weight of 0.1, and each
    class\ 2 sample is assigned a weight of 0.01. Sampling with this
    weight distribution, both classes have a 50% chance of being
    sampled.

[^note:cac_delta]: We compute change as the raw difference in accuracy
    (expressed as a fraction between 0 to 1). So, if the baseline model
    has an accuracy of 50%, and AlphaNet has an accuracy of 60%, the
    difference is $0.6 - 0.5 = 0.1$.
