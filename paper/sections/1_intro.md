# Introduction {#sec:intro}

Many AI systems are biased due to underlying biases in training data, in the
models themselves, or in the choices of the system designers [@2022.Ferrante.Lara].
Here we focus on the first of three sources of bias, observing that the
prevalence of objects in the real world follows a long-tailed distribution,
where some classes occur frequently and others occur rarely [@2014.Ramanan.Zhu;
 @2019.Yu.Liu]. Due to this phenomenon, many applications rely on models that
can classify rarer objects alongside common ones. For example, autonomous
vehicle systems are expected to classify rare animals, objects, or road
configurations in order to avoid potential collisions [@2019.Lucey.Chang] and
medical image analysis systems are used to spot rare cancers, detect anatomical
irregularities, and perform image reconstruction [@2022.Ferrante.Lara].

<!-- ![Part (a) shows the number of training samples per class for the CIFAR-100-LT
dataset, along with the per-class accuracy for a state of the art
model [@2020.Yu.Wang]. The plot shows the problem of performance imbalance with
long-tail learning, where class performance also follows a long-tail
distribution. Part (b) shows the high level idea behind our solution to this
problem. AlphaNet adaptively adjusts weak classifiers through nearest neighbor
compositions.](figures/teaser.pdf){#fig:teaser} -->

![Per-class test accuracy of the cRT model for classes in the 'few' split of
ImageNet-LT, versus the mean Euclidean distance to five nearest neighbors in the
'base' split.  Distance between classes was calculated using the mean
representation of training data samples. The plot shows a strong correlation
(Pearson correlation, $r$ shown in top right) between class test accuracy and
the mean nearest neighbor distance, that is, 'few' split classes with close
neighbors in the 'base' split had poor test
accuracy.](figures/cls_acc_vs_nndist_imagenetlt_crt_baseline){#fig:cls_acc_vs_nndist_baseline}

The significance of long-tailed distributions in real-world applications has
spurred a variety of approaches for long-tailed classification. Some methods
re-sample more data for less frequent classes in an effort to address data
imbalances [@2019.Belongie.Cui; @2019.Ma.Cao] [re-sampling can be challenging or
impossible, e.g., in medical contexts, @2022.Ferrante.Lara]. Other methods
re-weight the base classifiers so that rarer classifiers are weighted more
heavily and, thus, are better at classification [@2019.Kalantidis.Kang]. Both
re-sampling and re-weighting methods are strong baselines. State-of-the-art
results are achieved by more complex and difficult methods that rely on multiple
expert learning [@2020.Yu.Wang; @2021.Hwang.Cai], multi-stage
distillation [@2021.Wu.Li], or a large combination of several balancing,
re-sampling, and loss decays [@2022.Kong.Alshammari].

Although the above methods successfully improve long-tail performance, they
present several challenges that we address in our simple model. First, these
methods require re-training classifiers and thus are unable to be used in
applications where an existing frequent base classifier cannot be altered. For
example, some applications deal directly with privacy concerns or intellectual
property rights. Second, although these methods improve accuracy for rare
classes, they focus on also improving accuracy across _all_ classes within a
long-tailed distribution. This emphasis on overall accuracy leads the majority
of these long-tailed learning methods to suffer from a continued imbalance in
accuracy. For example, one highly effective baseline method,
cRT [@2019.Kalantidis.Kang] has an average accuracy of 61.8% on frequent classes,
but the average accuracy on rare classes is more than 30 points lower, at 27.4%.
Such imbalances can be particularly problematic in contexts where ethical
considerations come into play, as in medical image computing [@2022.Ferrante.Lara].
In @fig:teaser, we illustrate that, much as the classes themselves, _accuracies_
across classes follow a long-tailed distribution and that this distribution
correlates with class frequency. That is, most long-tail methods focus on data
balancing but fail to address accuracy balancing. Here, we argue that accuracy
balancing is equally crucial to long-tailed learning.

We address these two challenges by reducing the accuracy differences between
common classes and rarer classes -- critically _without_ modifying the existing
common classifiers that are the majority of the model's classifiers. Our goal is
to balance a set of imbalanced class accuracies in an existing model and _not_
to improve upon state-of-the-art. For example, in medical image computing,
outcome disparities (imbalanced class accuracies) have been connected to a "lack
of diversity and proper representation of the target population in the training
databases" [i.e., some classes being much rarer than others, @2022.Ferrante.Lara]. To
reduce accuracy differences, we propose AlphaNet, a simple method that transfers
knowledge from strong classifiers (i.e., learned with frequent data) to weak
classifiers (i.e., learned with rare data). AlphaNet, achieves more balanced
accuracy across classes by strengthening weak classifiers using a shallow
network to learn how to linearly combine each weak classifier with relevant
strong classifiers. To combine both relevant and non-relevant classifiers, our
methods allows for both positive and negative weights respectively. For example,
in @fig:teaser, we illustrate how the weak 'cat' classifier can be appropriately
adjusted by moving towards relevant strong 'leopard' and 'lion' classifiers and
away from non-relevant strong 'ostrich' classifier. Relevant strong classifiers
are simply determined by finding the nearest neighbor classes to the target
class (e.g., 'cat'). We emphasize that our method is a _post-hoc_ accuracy
correction that can be applied on top of any trained model that includes a set
of classifiers. Importantly, the adjustments we make to improve classifier
balance can be applied without touching the base classifiers. As such, our
method is applicable to use cases where existing base classifiers are either
unavailable or fixed (e.g., due to commercial interests or data privacy
protections). In these instances our method can uniquely provide accuracy
balancing updates.

To showcase the efficacy of our method, we applied it on top of strong baselines
and approaches for ImageNet-LT, Places-LT, and CIFAR-100-LT. Across these tests, our
improvement on rare classes was as high as 10 points. Applying AlphaNet on the
strong baselines described by @2019.Kalantidis.Kang, rare class accuracy was
increased by up to 5 points, while overall accuracy remained within 1 point,
showing that our approach provides a simple _post-hoc_ adjustment for balancing
class accuracies.

Summary of contributions:

- AlphaNet is a frustratingly simple _post-hoc_ method that can be applied on
  top of any classification model with a set of unique classifiers.

- The simplicity of our method lends to two additional advantages: 1) extremely
  rapid training (e.g. 5 minutes training for ImageNet-LT); 2) light weight
  computation that can be performed on a single small GPU or even CPU.

- We can strengthen weak classifiers without changing a large set of strong
  classifiers.

- Our method is robust, as shown by our trivial errors across multiple
  repetition runs.
