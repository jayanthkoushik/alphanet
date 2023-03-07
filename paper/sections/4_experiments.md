# Experiments {#sec:experiments}

## Experiment Setup

### Datasets

We evaluated our method using three long-tailed benchmark datasets:
ImageNet-LT, Places-LT\ [@2019.Yu.Liu], and CIFAR-100-LT\
[@2022.Kong.Alshammari]. These Datasets are sampled from their
respective original datasets, ImageNet\ [@2015.Fei-Fei.Russakovsky],
Places365\ [@2017.Torralba.Zhou], and
CIFAR-100\ [@2009.Hinton.Krizhevsky] such that the new distributions
follow a standard long-tailed distribution.  ImageNet-LT contains 1000
classes with the number of samples per class ranging from 5 to 4980
images. Places-LT contains 365 classes with the number of samples per
class ranging from 5 to 1280 images.  CIFAR-100-LT contains 100 classes
with the number of samples per class ranging from 5 to 500 images. For
CIFAR-100-LT, we used the version described by @2022.Kong.Alshammari,
using an imbalance factor of 100.

The datasets are broken down into three broad splits that indicate the
number of training samples per class: 1) 'many' contains classes with
greater than 100 samples; 2) 'medium' contains classes with greater than
or equal to 20 samples but less than or equal to 100 samples; 3) 'few'
contains classes with less than 20 samples. The test set is always
balanced, containing an equal number of samples for each class. We use
the term 'base' split to refer to the combined 'many' and 'medium'
splits.

Note: Another popular dataset used for testing long-tail learning models
is iNaturalist\ [@2018.Belongie.Horn]. Results for this dataset,
however, are much more balanced across classes. So it does not represent
a valid use case for our proposed method, and we omitted the dataset
from our experiments.

### Training data sampling

In order to prevent over-fitting on the 'few' split samples, we used a
class balanced sampling approach, using all 'few' split samples, and a
portion of the 'base' split samples. Given $F$ 'few' split samples, and
a ratio $\rho$, every epoch, $\rho F$ samples were drawn from the 'base'
split, with sample weights inversely proportional to the class size.
This ensured that all 'base' classes had an equal probability of being
sampled. As we show in the following section, $\rho$ allows us to
control the balance between 'few' and 'base' split accuracy.  We
evaluated AlphaNet with a range of $\rho$ values; results for
$\rho=0.5$, $\rho=1$, and $\rho=1.5$ are shown in @sec:exp:res, and the
full set of results is in the appendix.

### Training

All experiments used an AlphaNet module with three 32 unit layers, and
Leaky-ReLU\ [@LEAKYRELU] activation. Unless stated otherwise, euclidean
distance was used to find $k=5$ nearest neighbors for each 'few' split
class. Models were trained for 25 epochs to minimize cross-entropy loss
computed using mini-batches of 64 samples. Optimization was performed
using AdamW\ [@2017.Hutter.Loshchilov] with a learning rate of 0.001,
decayed by a factor of 10, every 10 epochs. Model weights were saved
after each epoch, and after training, the weights with the best accuracy
on validation data were used to report results on the test set.  All
experiments were repeated 10 times, and we report mean and standard
deviation of accuracies across trials.

## Results {#sec:exp:res}

### Baseline models

First, we applied AlphaNet on several strong baselines proposed by
@2019.Kalantidis.Kang. These methods have good overall accuracy, but
accuracy for 'few' split classes is much lower. On the ImageNet-LT
dataset, average accuracy for the cRT and LWS models (using a ResNeXt-50
backbone) is nearly 20 points below the overall accuracy, as seen in
@tbl:imgnetlt_baselines. Using features extracted from these two models,
we used AlphaNet to update 'few' split classifiers. On both models, we
saw a significant increase in the 'few' split accuracy for all values of
$\rho$, leading to more balanced accuracies. For $\rho = 1$, average
'few' split accuracy was boosted by 7 points for the cRT model, and
about 11 points for the LWS model, while overall accuracy was within 2
point of the original. We found this value of $\rho$ to provide a good
balance between 'few' split and overall accuracy. Lower values of $\rho$
led to much larger 'few' split accuracy, and even with $\rho = 1.5$,
there was a significant increase, while the accuracy for other splits
remained around the same.

{% include tables/imagenetlt_baselines.md %}

We repeated the above experiment on the Places-LT dataset, where again
'few' split accuracy for the cRT and LWS models is much lower than the
overall accuracy (by around 12 and 9 points respectively as seen in
@tbl:placeslt_baselines. With $\rho = 1$, AlphaNet improved 'few' split
accuracy by 2 points on average for the cRT model, and about 6 points on
average for the LWS model. In both cases, overall accuracy was within 1
point of the baseline.

{% include tables/placeslt_baselines.md %}

### State-of-the-art

Next, we applied AlphaNet on two state-of-the-art models. First, we used
the 6-expert teacher model of @2020.Yu.Wang. We provided the combined
feature vectors from all 6 experts as input to AlphaNet. The learned
'few' split classifiers were split into 6, and used to update the
experts. Prediction scores from the experts were averaged to produce the
final predictions, as in the original model. For ImageNet-LT, the
experts used a ResNeXt-50 backbone, and for CIFAR-100-LT, a ResNet-32
backbone. @tbl:ride_imagenetlt_placeslt_short shows the base results for
the expert models, along with AlphaNet results for $\rho = 0.5, 1, 2$.
On ImageNet-LT, 'few' split accuracy was increased by up to 7 points,
and on CIFAR-100-LT, by 5 points. The second state-of-the-art model we
used was the weight balancing method proposed by @2022.Kong.Alshammari.
The full set of results on the CIFAR-100-LT dataset is shown in the
appendix.

{% include tables/ride_imagenetlt_placeslt_short.md %}

### Comparison with control

Our method is based on the core hypothesis that classifiers can be
improved using nearest neighbors. In this section, we directly test this
hypothesis.  Based on the results in the previous section, the
improvements in 'few' split accuracy could be attributed simply to the
extra fine-tuning of 'few' split classifiers. So, we repeated the
experiments of the previous section with randomly chosen neighbors for
each 'few' split class, rather than nearest neighbors. This differs from
our previous experiments only in the nature of neighbors used, so if our
method's improvements are solely due to extra fine-tuning, we should see
the same results. However, as seen in
@fig:euclidean_random_split_deltas_vs_imagenetlt_crt, training with
nearest neighbors garners much larger improvements in 'few' split
accuracy, with similar trends in overall accuracy.  This supports our
hypothesis that data-poor classes can make use of data from neighbors to
improve classification performance.

![Change in split accuracy for AlphaNet training of cRT model on
ImageNet-LT.  For each value of $\rho$, the two plots both show the
change in split accuracy for AlphaNet compared to the baseline cRT
model, as well as the change in overall accuracy. Plot a) shows the
results for normal training with 5 euclidean nearest neighbors, and plot
b) shows the results for training with 5 random neighbors for each 'few'
split class. Training with nearest neighbors leads to much larger
increase in 'few' split accuracy (especially for small values of
$\rho$), which cannot be accounted for by the additional fine-tuning of
classifiers.](figures/euclidean_random_split_deltas_vs_rho_imagenetlt_crt){#fig:euclidean_random_split_deltas_vs_imagenetlt_crt}

![Change in per-class test accuracy for AlphaNet applied over cRT
features, with $\rho=0.5$.  Within each split, the classes are sorted by
change in accuracy, and the dashed line shows the average per-class
change (change value $\Delta$).  The spanning dashed line shows the mean
overall change in per-class test accuracy. Nearly all 'few' split
classes have their accuracy improved, and the overall improvement of
'few' split accuracy far exceeds the average decrease for 'many' and
'medium'
splits.](figures/cls_deltas_imagenetlt_crt_rho_05){#fig:cls_deltas}

...

<div id="fig:cls_delta_vs_nndist">

![Five nearest neighbors by Euclidean
distance](figures/cls_delta_vs_nndist_imagenetlt_crt_rho_05){#fig:cls_delta_vs_nndist_euclidean}

![Five random
neighbors](figures/cls_delta_vs_nndist_imagenetlt_crt_randomnns_rho_05){#fig:cls_delta_vs_nndist_random}

Change in per-class accuracy for AlphaNet applied over cRT features,
with $\rho=0.5$, versus mean Euclidean distance to five neighbors.
@fig:cls_delta_vs_nndist_euclidean uses five nearest neighbors by
Euclidean neighbors, and @fig:cls_delta_vs_nndist_random uses five
random neighbors.  Both plots show results from ten runs with random
starts. Comparing with @fig:cls_acc_vs_nndist_baseline, we can see that
AlphaNet provides the largest boost to classes with poor baseline
performance, which have close nearest neighbors.
</div>

...

![Change in 'few' split sample predictions for AlphaNet applied for cRT
features, with $k=10$ nearest neighbors by Euclidean distance. In each
chart, the bars on the left show the distribution of predictions for
'few' split test samples, by the baseline model; and the bars on the
right show the distribution for AlphaNet. The predictions are grouped
into three categories: 1) correct predictions, 2) incorrect predictions
where the prediction was a nearest neighbor of the original class (e.g.,
predicting 'Malamute' for 'Husky'), and 3) all other incorrect
predictions. The "flow" bands from left to right show the changes in
individual sample predictions. There is a large improvement for samples
previously misclassified as a nearest neighbor, particularly for small
$\rho$. As $\rho$ is increased, smaller portion of these mistakes are
corrected, leading to smaller improvement in 'few' split
accuracy.](figures/rhos_pred_changes_imagenetlt_crt_k_10){#fig:rhos_pred_changes}

...

<!-- ### Relation to nearest neighbors

We tested the effect of $k$, the number of nearest neighbors per 'few' split
class, under both Euclidean and cosine distance. @fig:split_acc_vs_k shows the
per-split and overall accuracy for AlphaNet trained on the cRT model for
ImageNet-LT with $\rho=0.5$, and $k = 1,\dots,10$. Results for other values of
$\rho$ are shown in the appendix. As we can see, both Euclidean and cosine distance
produced similar results, and per-split accuracies plateaued around $k=5$.

![Per-split test accuracy for AlphaNet training of cRT model on ImageNet-LT,
versus the number of nearest neighbors used for each 'few' split class. There is
little change in accuracies beyond $k=5$.](figures/split_acc_vs_k){#fig:split_acc_vs_k} -->

<!-- ![alphas](figures/alphas.pdf){#fig:alphas} -->

<!--
### Analysis of Per-class Accuracies

Let us now take a closer look at the improvements provided by AlphaNet.
@fig:per_class_diffs shows the change in ImageNet-LT per-class accuracy on using
AlphaNet with the cRT model. We can see that nearly all 'few' split classes have
their accuracy significantly increased. Any decrease in accuracy for other
classes is far less in comparison, leading to overall accuracy remaining almost
the same, despite the number of 'few' split classes being much less than the
number of 'base' classes. TODO: should we add the dummy model results here to
show that we can't achieve these results just by predicting 'few' split classes
more often?

![Change in per-class accuracy for AlphaNet applied on cRT model. Each
colored segment is a different split, ordered 'many', 'medium', and
'few' left to right. The dotted line shows the average change in
per-class accuracy. We see that AlphaNet significantly improves the
accuracy of 'few' split classes, while the overall change is close to
0.](figures/imagenetlt_resnext50_crt_delta_accs){#fig:per_class_diffs}

### Analysis on Rebalanced Accuracies

As we have seen, models on long-tail datasets suffer from accuracies themselves
showing a long-tail distribution, and class accuracy being inversely
proportional to the number of samples from the class. In @fig:trade_off, we show
that AlphaNet directly addresses this issue, and reduces the inverse relation
between samples and accuracy. Furthermore, this reduction can be controlled by
the parameter $\rho$. Larger values of $\rho$ lead to less dependence of
accuracy on data, leading to more balanced accuracies across classes.

![Linear regression of class accuracy versus log of class samples. Each
line represents a different value of $\rho$, and the shaded bands are
bootstrapped confidence intervals from 10 independent
runs.](figures/imagenetlt_resnext50_crt_acc_vs_samples){#fig:trade_off}
-->
