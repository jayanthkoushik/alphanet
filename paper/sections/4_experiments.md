# Experiments {#sec:exp}

## Experimental setup {#sec:exp:setup}

**Datasets.** We evaluated our method using three long-tailed datasets:
ImageNet-LT, Places-LT\ [@2019.Yu.Liu], and CIFAR-100-LT\ [@2022.Kong.Alshammari].
These Datasets are sampled from their respective original datasets,
ImageNet\ [@2015.Fei-Fei.Russakovsky], Places365\ [@2017.Torralba.Zhou], and
CIFAR-100\ [@2009.Hinton.Krizhevsky] such that the new distributions follow a
standard long-tailed distribution. For CIFAR-100-LT, we used an imbalance
factor of 100.

The datasets are broken down into three broad splits that indicate the number
of training samples per class: 1) 'many' contains classes with greater than 100
samples; 2) 'medium' contains classes with greater than or equal to 20 samples
but less than or equal to 100 samples; 3) 'few' contains classes with less than
20 samples. The test set is always balanced, containing an equal number of
samples for each class. We use the term 'base' split to refer to the combined
'many' and 'medium' splits.

Note: Another popular dataset used for testing long-tail models is
iNaturalist\ [@2018.Belongie.Horn]. Results for this dataset, however, are much
more balanced across splits. So it does not represent a valid use case for our
proposed method, and we omitted the dataset from our experiments.

**Training data sampling.** In order to prevent over-fitting on the 'few' split
samples, we used a class balanced sampling approach, using all 'few' split
samples, and a portion of the 'base' split samples. Given $F$ 'few' split
samples, and a ratio $\rho$, every epoch, $\rho F$ samples were drawn from the
'base' split, with sample weights inversely proportional to the class size.
This ensured that all 'base' classes had an equal probability \aarti{base
classes don't have exactly same size so why do you have equal probability?} of
being sampled. As we show in the following section, $\rho$ allows us to control
the balance between 'few' and 'base' split accuracy. We evaluated AlphaNet with
a range of $\rho$ values; results for $\rho=0.5$, $\rho=1$, and $\rho=1.5$ are
shown in the following section, and the full set of results is in the appendix.

**Training.** All experiments used an AlphaNet module with three 32 unit
layers, and Leaky-ReLU activation\ [@2016.Farhadi.Redmon]. Unless stated
otherwise, euclidean distance was used to find $k=5$ nearest neighbors for each
'few' split class. Models were trained for 25 epochs to minimize cross-entropy
loss computed using mini-batches of 64 samples. Optimization was performed
using AdamW\ [@2017.Hutter.Loshchilov] with a learning rate of 0.001, decayed
by a factor of 10, every 10 epochs. Model weights were saved after each epoch,
and after training, the weights with the best accuracy on validation data were
used to report results on the test set. All experiments were repeated 10 times,
and we report mean and standard deviation of accuracies across trials.

## Long-tail classification results {#sec:exp:classres}

{% include tables/imagenetlt_placeslt_baselines.md %}

**Baseline models.** First, we applied AlphaNet on several strong baseline
methods\ [@2019.Kalantidis.Kang] \aarti{call them out - its not just cRT and
LWS but also cross entropy, NCM (include full form) and $\tau$-normalized with
references}. These methods have good overall accuracy, but accuracy for 'few'
split classes is much lower. On the ImageNet-LT dataset, average accuracy for
the \ac{cRT} and \ac{LWS} models (using a ResNeXt-50 backbone) is nearly 20
points below the overall accuracy, as seen in @tbl:baselines_imagenetlt_placeslt.
Using features extracted from these two models, we used AlphaNet to update
'few' split classifiers. On both models, we saw a significant increase in the
'few' split accuracy for all values of $\rho$. For $\rho = 1$, average 'few'
split accuracy was boosted by 7 points for the \ac{cRT} model, and about 11
points for the \ac{LWS} model.

We repeated the above experiment on the Places-LT dataset, where again 'few'
split accuracy for the cRT and LWS models is much lower than the overall
accuracy (by around 12 and 9 points respectively as seen in
@tbl:baselines_imagenetlt_placeslt. With $\rho = 1$, AlphaNet improved 'few'
split accuracy by 2 points on average for the cRT model, and about 6 points on
average for the LWS model. \aarti{Also comment on overall accuracies - are
these with grouping semantically similar classes as discussed in intro?}

{% include tables/imagenetlt_cifarlt_ride_short.md %}

**Expert model.** Next, we applied AlphaNet on the 6-expert ensemble \ac{RIDE}
model\ [@2020.Yu.Wang]. We provided the combined feature vectors from all 6
experts as input to AlphaNet. The learned 'few' split classifiers were split
into 6, and used to update the experts. Prediction scores from the experts were
averaged to produce the final predictions, as in the original model. For
ImageNet-LT, the experts used a ResNeXt-50 backbone, and for CIFAR-100-LT, a
ResNet-32 backbone. @tbl:ride_imagenetlt_cifarlt_short shows the base results
for the expert models, along with AlphaNet results for $\rho = 0.5, 1, 2$. On
ImageNet-LT, 'few' split accuracy was increased by up to 7 points, and on
CIFAR-100-LT, by 5 points.

## Comparison with control {#sec:exp:control}

![Change in split accuracy for AlphaNet training of cRT model on ImageNet-LT.
For each value of $\rho$, the two plots show the change in split accuracy for
AlphaNet compared to the baseline cRT model, as well as the change in overall
accuracy. Left shows the results for normal training with 5 euclidean nearest
neighbors, and right shows the results for training with 5 random neighbors for
each 'few' split class. Training with nearest neighbors leads to a larger
increase in 'few' split accuracy (especially for small values of $\rho$), which
cannot be accounted for by the additional fine-tuning of
classifiers.](figures/euclidean_random_split_deltas_vs_rho_imagenetlt_crt){#fig:euclidean_random_split_deltas_vs_imagenetlt_crt}

Our method is based on the core hypothesis that classifiers can be improved
using nearest neighbors. In this section, we directly test this hypothesis.
Based on the results in the previous section, the improvements in 'few' split
accuracy could be attributed simply to the extra fine-tuning of 'few' split
classifiers. So, we repeated the experiments of the previous section with
randomly chosen neighbors for each 'few' split class, rather than nearest
neighbors. This differs from our previous experiments only in the nature of
neighbors used, so if our method's improvements are solely due to extra
fine-tuning, we should see the same results. However, as seen in
@fig:euclidean_random_split_deltas_vs_imagenetlt_crt, training with nearest
neighbors garners much larger improvements in 'few' split accuracy, with
similar trends in overall accuracy. This supports our hypothesis that data-poor
classes can make use of information from neighbors to improve classification
performance.

## Prediction changes {#sec:exp:predchanges}

![Change in per-class accuracy for AlphaNet applied over cRT features, versus
mean Euclidean distance to five nearest neighbors. Comparing with
@fig:analysis:acc_vs_dist, we can see that AlphaNet provides the largest boost
to classes with poor baseline performance, which have close nearest
neighbors.](figures/cls_delta_vs_nndist_imagenetlt_crt_rho_05){#fig:cls_delta_vs_nndist}

As shown in @sec:intro, the \ac{cRT} model frequently misclassifies 'few' split
classes as visually similar 'base' split classes. Using the AlphaNet model with
$\rho=0.5$, we performed the same analyses as before. @fig:pred_changes:few
shows the change in sample predictions, A large portion of samples previously
misclassified as a nearest neighbor are correctly classified after their
classes are updated with AlphaNet. Furthermore, as seen in
@fig:cls_delta_vs_nndist, AlphaNet improvements are strongly correlated to mean
nearest neighbor distance. Classes with close neighbors, which had a high
likelihood of being misclassified by the baseline model, see the biggest
improvement in test accuracy.

## Analysis of AlphaNet predictions {#sec:exp:analysis}

<div id="fig:pred_changes">

![Predictions on 'few' split classes, with nearest neighbors selected from
'base' split
classes.](figures/pred_changes_imagenetlt_crt_rho_05_few){#fig:pred_changes:few}

![Predictions on 'base' split classes, with nearest neighbors selected from
'few' split
classes.](figures/pred_changes_imagenetlt_crt_rho_05_base){#fig:pred_changes:base}

![All predictions, with nearest neighbors selected from all
classes.](figures/pred_changes_imagenetlt_crt_rho_05_all){#fig:pred_changes:all}

Change in sample predictions for AlphaNet applied for \ac{cRT} features, with
$k=10$ nearest neighbors by Euclidean distance. The bars on the left show the
distribution of predictions by the baseline model; and the bars on the right
show the distribution for AlphaNet. The counts are aggregated from 10 different
runs of AlphaNet. The "flow" bands from left to right show the changes in
individual sample predictions.

</div>

AlphaNet significantly boosts the accuracy of 'few' split classes. However,
looking at @tbl:baselines_imagenetlt_placeslt and
@tbl:ride_imagenetlt_cifarlt_short, we see that the overall accuracy decreases
compared to baseline models, particularly for small values of $\rho$. It is
important to note that the increase in 'few' split accuracy is much larger than
the decrease in overall accuracy. As discussed earlier, in many applications it
is important to have balanced performance across classes, and AlphaNet succeeds
in making accuracies more balanced across splits.

However, we further analyzed the prediction changes for 'base' split samples.
Specifically, @fig:pred_changes:base shows change in predictions for 'base'
split samples, with nearest neighbors selected from the 'few' split. We see a
small increase in misclassifications as 'few' split classes. This leads to the
slight decrease in overall accuracy, which is also evident in
@fig:pred_changes:all where all predictions are shown, and with nearest
neighbors from all classes.

<div id="fig:pred_changes_semantic">

![Predictions on 'few' split classes, with nearest neighbors selected from
'base' split
classes.](figures/pred_changes_imagenetlt_crt_rho_05_semantic_few){#fig:pred_changes_semantic:few}

![Predictions on 'base' split classes, with nearest neighbors selected from
'few' split
classes.](figures/pred_changes_imagenetlt_crt_rho_05_semantic_base){#fig:pred_changes_semantic:base}

![All predictions, with nearest neighbors selected from all
classes.](figures/pred_changes_imagenetlt_crt_rho_05_semantic_all){#fig:pred_changes_semantic:all}

Change in sample predictions for AlphaNet applied for \ac{cRT} features,
with nearest neighbors identified using WordNet categories. This figure
represents the same predictions as @fig:pred_changes, but grouped
differently.

</div>

The previous analysis was conducted using nearest neighbors identified based on
visual similarity. Since this is dependent on the particular model, we
conducted an additional analysis to see the behavior of predictions with
respect to _semantically similar_ categories. For classes in ImageNet-LT, we
defined nearest neighbors using distance in the WordNet hierarchy.
Specifically, if two classes (e.g., 'Lhasa' and 'Tibetan terrier') share a
parent at most four levels higher in WordNet (in this example, 'dog'), we
consider them to be nearest neighbors. @fig:pred_changes_semantic shows the
predictions for AlphaNet with cRT grouped based on these nearest neighbors. As
we can see, a large number of predictions which are considered incorrect are
among semantically similar categories which can be hard for even humans to
distinguish. This suggests that metrics for long-tail classification might need
to be re-evaluated for large datasets with many similar classes.