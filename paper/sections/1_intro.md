# Introduction {#sec:intro}

Objects in the real world follow a long-tailed distribution, where many
categories occur rarely[@long.tail.def]. Due to this phenomenon, many
computer vision applications require models that can classify rarer
objects alongside common ones. For example, autonomous vehicle systems
are expected to classify rare animals, objects, or road configurations
in order to avoid potential collisions[@av.long.tail]; medical image
analysis systems should spot rare cancers, detect anatomical
irregularities, and perform image reconstruction[@medical.long.tail]. In
this work, we focus on object classification with long-tailed datasets,
i.e., long-tail classification.

The significance of long-tailed distributions in real-world applications
has spurred a variety of approaches for long-tail classification.
Learning in this setting is challenging because many classes have a
small number of samples. Some methods re-sample more data for infrequent
classes in an effort to address data imbalance[@resampling.examples].
Other methods adjust learned classifiers to re-weight (TODO: should it
be 'weight' or 'weigh') them in favor of rare
classes[@reweigh.examples]. Both re-sampling and re-weighting based
methods are strong baselines on long-tail classification tasks.
State-of-the-art results are achieved by more complex methods that, for
example, learn multiple experts[@experts.examples], perform multi-stage
distillation[@distill.examples], or use a combination of weight
balancing, data re-sampling, and loss decay[@combo.examples].

Despite advances in the field, accuracy on rare classes continues to be
significantly worse than overall accuracy. For example, on the
ImageNet-LT dataset, the expert model of @ride has an average accuracy
of xx.x% on frequent classes, but the accuracy on rare classes is more
than 30 points lower, at xx.x%. In addition to significantly reducing
overall accuracy, the performance imbalance raises ethical concerns in
contexts where such behavior leads to biased outcomes, for instance as
in medical image computing[@medical.bias.concern] (TODO: find other
instances of bias). For these reasons, we focus in particular on
improving the accuracy for rare classes in long-tail classification.

## Analysis {#sec:intro:analysis}

**Notation**: Throughout this work, we will define the distance
between two classes as the distance between their average training set
representation. Given a classification model, let $f$ be the function
mapping images to vectors in $\R^d$ (typically, this is the output of
the penultimate layer in convolutional networks). For a class $c$ with
training samples $I^c_1, \dots, I^c_{n_c}$, let $x^c \equiv (1/n_c)
\sumnl_i f(I^c_i)$. Given a distance metric $\mu: \R^d \times \R^d \to \R$,
for classes $c_1$ and $c_2$, we define $m_\mu(c_1, c_2) \equiv
\mu(x^{c_1}, x^{c_2})$.

To understand the poor rare class performance of long-tail models, we
analyzed the predictions of the \ac{RIDE} model[@ride] on 'few' split
test samples in ImageNet-LT. The 'few' split is comprised of classes
with less than 20 training samples (note that at test time, all classes
have 50 samples), and as noted earlier, the \ac{RIDE} model achieves an
accuracy of xx.xx% on this split. To categorize predictions, for each
class in the 'few' split of ImageNet-LT, we found the 10 nearest
neighbors in the 'base' split (all classes not in the 'few' split) using
Euclidean distance ($\mu(a, b) = \norm{a - b}$) between features from
the \ac{RIDE} model. For any class, we will refer to its nearest
neighbors as visually similar classes. @fig:analysis:bins shows
predictions on the test set binned into three groups: 1) samples
predicted correctly, 2) samples incorrectly predicted as a visually
similar class (e.g., predicting 'husky' instead of 'malamute'), and 3)
samples incorrectly predicted as a visually dissimilar class (e.g.,
predicting 'camel' instead of 'malamute'). We can see that a significant
portion of the misclassifications (about xx% in this case) are to
visually similar classes. @fig:analysis:egs shows samples from one pair
of visually similar classes; the differences are subtle, and can be hard
even for humans to identify.

To make the influence of visually similar classes concrete, we next
analyzed the relationship between test accuracy and mean nearest
neighbor distance (i.e., the average $m_\mu$ between a class and its
nearest neighbors). This is shown in @fig:analysis:acc_vs_dist, where
we can see a strong positive correlation between accuracy and mean
distance--'few' split classes with close neighbors have smaller test
accuracy than classes with distant neighbors.

## Method overview {#sec:intro:overview}

Based on the previous analysis, we designed a method, AlphaNet, to
improve classifiers for rare classes using information from visually
similar frequent classes. We will use the term 'classifier' to denote
the linear mapping from feature vectors to class scores. For a class
$c$, this is a vector $w_c \in R^d$ (in convolutional networks, the last
layer is generally a matrix of all individual classifiers). Given a
feature vector $z = f(I)$ for some image $I$, $w_c^T z$ is the
prediction score for class $c$ (the bias term is omitted here for
simplicity), and the model's class prediction for $I$ is given by
$\argmax_c w_c^T f(I)$.

@fig:pipeline shows the overview of our method. At a high level,
AlphaNet can be seen as moving the classifiers for rare classes based on
their position relative to visually similar classes. Importantly, it
updates classifiers without making any changes to the
representation space, or to other classifiers in the model. It performs
a post-hoc correction, and as such, is applicable to use cases where
existing base classifiers are either unavailable or fixed (e.g., due to
commercial interests or data privacy protections). The simplicity of
our method lends to computation advantages--AlphaNet can be trained
rapidly, and without need for multiple \acp{GPU}.

<!-- OUTLINE -->

<!-- 1) Real-world classes follow a long-tail distribution, and datasets with
balanced classes are not representative. 2) Long-tail learning aims to
learn from long-tail datasets, such as ImageNet-LT. 4) A big challenge
in long-tail learning is getting good performance on rare classes, which
might have very limited samples. -->

<!-- 1) There has been a lot of research on long-tail learning, which have
been continuously raising the test accuracy. 2) RIDE is a
state-of-the-art method which achieves an accuracy of <overall_acc> on
ImageNet-LT. 3) However, performance on the 'few' split, which are
classes with less than 20 samples during training, is significantly
worse, with an accuracy of <few_acc>. 4) In many applications, it is
important for accuracy on data-poor classes to be at par with the
overall accuracy, and for models to not be biased. 5) So, specifically
improving 'few' split accuracy is important to consider in long-tail
learning, in addition to overall accuracy. -->

<!-- 1) Let us look at the 'few' split results of RIDE in more detail, by
analyzing the predictions on test set samples. 2) First, we find the 10
nearest neighbors from 'base' split for every 'few' split class. 3) In
figure <analysis:a> we show the predictions, grouped into a) correct, b)
incorrect as a nearest neighbor, and c) incorrect as a distant class. 4)
A significant number of misclassifications are of the second category,
i.e., many 'few' split samples are mistakenly classified as close
neighbors of the true class. 5) Furthermore, as seen in figure
<analysis:b>, test-accuracy is directly proportional to the mean
distance to nearest neighbors. -->

<!-- 1) Based on the previous analysis, we designed a method, AlphaNet, to
improve 'few' split classifiers using information from nearest neighbors
in the 'base' split. (In general we improve 'few' split accuracy, we see
our greatest improvement because...). 2) At a high level, our method can
be seen as moving the 'few' split classifiers based on their position
with respect to similar visual classes. 3) We update each 'few' split
classifier using a linear combination of classifiers from nearest
neighbors, with the combination weights learned using a shallow neural
network. 3) As seen in figure <pipeline>, our method is simple to use,
and requires very little training. 4) Our method can be applied on any
existing model, and performs only a _post hoc_ correction, without
modifying the learned representations, or the 'base' split classifiers. -->

<!-- The rest of the paper will be structured as follows (TODO: do we need this?):

* Related work
* Description of method
* Experimental results on long-tail datasets
* Analysis of AlphaNet predictions compared to baseline model predictions
* Evaluation of AlphaNet using WordNet categories
* Discussion -->

<!-- SKELETON -->

<!-- 1. Intro to long-tail learning
   1.1. Need for long-tail learning
   1.2. Description and examples -->

<!-- 2. State of long-tail results
   2.1. Recent state of the art method, and its performance
   2.2. Few-split accuracy -->

<!-- 3. Analysis of few-split accuracy
   3.1. Break-down of test predictions on ImageNet-LT
   3.2. Categorization of test predictions based on nearest neighbors
   3.3. Relation between test accuracy and closeness to nearest neighbors

4. Motivation behind AlphaNet
   4.1. Using NNs to adjust 'few'-split classifiers
   4.2. High level description of AlphaNet "pipeline"
   4.3. Usability of AlphaNet

5. Paper structure -->

<!-- OLD -->

<!-- Many AI systems are biased due to underlying biases in training data, in
the models themselves, or in the choices of the system designers
[@2022.Ferrante.Lara]. Here we focus on the first of three sources of
bias, observing that the prevalence of objects in the real world follows
a long-tailed distribution, where some classes occur frequently and
others occur rarely [@2014.Ramanan.Zhu; @2019.Yu.Liu]. Due to this
phenomenon, many applications rely on models that can classify rarer
objects alongside common ones. For example, autonomous vehicle systems
are expected to classify rare animals, objects, or road configurations
in order to avoid potential collisions [@2019.Lucey.Chang] and medical
image analysis systems are used to spot rare cancers, detect anatomical
irregularities, and perform image reconstruction [@2022.Ferrante.Lara]. -->

<!-- The significance of long-tailed distributions in real-world applications has
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
re-sampling, and loss decays [@2022.Kong.Alshammari]. -->

<!-- Although the above methods successfully improve long-tail performance, they
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
considerations come into play, as in medical image computing [@2022.Ferrante.Lara]. -->

<!-- We address these two challenges by reducing the accuracy differences between
common classes and rarer classes -- critically _without_ modifying the existing
common classifiers that are the majority of the model's classifiers. Our goal is
to balance a set of imbalanced class accuracies in an existing model and _not_
to improve upon state-of-the-art. For example, in medical image computing,
outcome disparities (imbalanced class accuracies) have been connected to a "lack
of diversity and proper representation of the target population in the training
databases" [i.e., some classes being much rarer than others,
@2022.Ferrante.Lara]. As seen in @fig:cls_acc_vs_nndist_baseline, accuracy on
data-poor classes is strongly linked to closeness to a nearest neighbor--classes
with a close neighbor are often mistakenly classified as the neighbor. To reduce
accuracy differences, we propose AlphaNet, a simple method that transfers
knowledge from strong classifiers (i.e., learned with frequent data) to weak
classifiers (i.e., learned with rare data). AlphaNet achieves more balanced
accuracy across classes by strengthening weak classifiers using a shallow
network to learn how to linearly combine each weak classifier with relevant
strong classifiers. To combine both relevant and non-relevant classifiers, our
methods allows for both positive and negative weights respectively. Relevant
strong classifiers are simply determined by finding the nearest neighbor classes
to the target class (e.g., 'cat'). We emphasize that our method is a _post-hoc_
accuracy correction that can be applied on top of any trained model that
includes a set of classifiers. Importantly, the adjustments we make to improve
classifier balance can be applied without touching the base classifiers. As
such, our method is applicable to use cases where existing base classifiers are
either unavailable or fixed (e.g., due to commercial interests or data privacy
protections). In these instances our method can uniquely provide accuracy
balancing updates. -->

<!-- To showcase the efficacy of our method, we applied it on top of strong baselines
and approaches for ImageNet-LT, Places-LT, and CIFAR-100-LT. Across these tests, our
improvement on rare classes was as high as 10 points. Applying AlphaNet on the
strong baselines described by @2019.Kalantidis.Kang, rare class accuracy was
increased by up to 5 points, while overall accuracy remained within 1 point,
showing that our approach provides a simple _post-hoc_ adjustment for balancing
class accuracies. -->

<!-- Summary of contributions:

- AlphaNet is a frustratingly simple _post-hoc_ method that can be applied on
  top of any classification model with a set of unique classifiers.

- The simplicity of our method lends to two additional advantages: 1) extremely
  rapid training (e.g. 5 minutes training for ImageNet-LT); 2) light weight
  computation that can be performed on a single small GPU or even CPU.

- We can strengthen weak classifiers without changing a large set of strong
  classifiers.

- Our method is robust, as shown by our trivial errors across multiple
  repetition runs. -->

<!-- ![Part (a) shows the number of training samples per class for the CIFAR-100-LT
dataset, along with the per-class accuracy for a state of the art
model [@2020.Yu.Wang]. The plot shows the problem of performance imbalance with
long-tail learning, where class performance also follows a long-tail
distribution. Part (b) shows the high level idea behind our solution to this
problem. AlphaNet adaptively adjusts weak classifiers through nearest neighbor
compositions.](figures/teaser.pdf){#fig:teaser} -->

<!-- In @fig:teaser, we illustrate that, much as the classes themselves, _accuracies_
across classes follow a long-tailed distribution and that this distribution
correlates with class frequency. That is, most long-tail methods focus on data
balancing but fail to address accuracy balancing. Here, we argue that accuracy
balancing is equally crucial to long-tailed learning. -->

<!-- ![Per-class test accuracy of the cRT model for classes in the 'few' split of
ImageNet-LT, versus the mean Euclidean distance to five nearest neighbors in the
'base' split. Distance between classes was calculated using the mean
representation of training data samples. The plot shows a strong correlation
(Pearson correlation, $r$ shown in top right) between class test accuracy and
the mean nearest neighbor distance, that is, 'few' split classes with close
neighbors in the 'base' split had poor test
accuracy.](figures/cls_acc_vs_nndist_imagenetlt_crt_baseline){#fig:cls_acc_vs_nndist_baseline} -->
