# Introduction {#sec:intro}

Objects in the real world follow a long-tailed distribution, where many
categories occur only rarely[@long.tail.def]. Due to this phenomenon,
many computer vision applications require models that can learn to
accurately classify rarer objects alongside common ones. For example,
autonomous vehicle systems are expected to classify rare animals,
objects, or road configurations in order to avoid potential
collisions[@av.long.tail]; medical image analysis systems should spot
rare cancers, detect unusual anatomical irregularities, and reconstruct
images from few examples[@medical.long.tail]. To address challenges in
these and related use cases, we focus on object classification using
long-tailed datasets -- "long-tail classification".

The significance of long-tailed distributions in real-world applications
has spurred a variety of approaches for long-tail classification.
Learning in this setting is challenging because many classes are "rare"
-- having only a small number of samples. Some methods re-sample more
data for rare classes in an effort to address data
imbalance[@resampling.examples]. Other methods adjust learned
classifiers to re-weight them in favor of rare
classes[@reweigh.examples]. Both re-sampling and re-weighting methods
provide strong baselines for long-tail classification tasks. However,
state-of-the-art results are achieved by more complex methods that, for
example, learn multiple experts[@experts.examples], perform multi-stage
distillation[@distill.examples], or use a combination of weight
balancing, data re-sampling, and loss decay[@combo.examples].

Despite these advances, accuracy on rare classes continues to be
significantly worse than overall accuracy. For example, on the
ImageNet-LT dataset, the expert model of @ride has an average accuracy
of xx.x% on frequent classes, but an average accuracy of xx.x% on rare
classes. In addition to significantly reducing overall accuracy, such
performance imbalances raise ethical concerns in contexts where unequal
accuracy leads to biased outcomes, for instance as in medical image
computing[@medical.bias.concern] (TODO: find other instances of bias).
For these reasons, our method is aimed at directly improving the
accuracy for rare classes in long-tail classification.

## Analysis {#sec:intro:analysis}

**Notation**: Throughout this work, we will define the distance between
two classes as the distance between their average training set
representation. Given a classification model, let $f$ be the function
mapping images to vectors in $\R^d$ (typically, this is the output of
the penultimate layer in convolutional networks). For a class $c$ with
training samples $I^c_1, \dots, I^c_{n_c}$, let $\v{x}^c \equiv
(1/n_c) \sumnl_i f(I^c_i)$. Given a distance metric $\mu: \R^d \times
\R^d \to \R$, for classes $c_1$ and $c_2$, we define $m_\mu(c_1, c_2)
\equiv \mu(\v{x}^{c_1}, \v{x}^{c_2})$.

To understand the poor rare class performance of long-tail models, we
analyzed the predictions of the \ac{RIDE} model[@ride] on 'few' split
test samples in ImageNet-LT. The 'few' split is comprised of classes
with less than 20 training samples (note that at test time, all classes
have 50 samples), and as noted earlier, the \ac{RIDE} model achieves an
accuracy of xx.xx% on this split. To categorize predictions, for each
class in the 'few' split of ImageNet-LT, we found the 10 nearest
neighbors in the 'base' split (all classes not in the 'few' split) using
Euclidean distance ($\mu(\v{a}, \v{b}) = \norm{\v{a} - \v{b}}$) between
features from the \ac{RIDE} model. For any class, we will refer to its
nearest neighbors as visually similar classes. @fig:analysis:bins shows
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
$c$, this is a vector $\v{w}_c \in R^d$ (in convolutional networks, the
last layer is generally a matrix of all individual classifiers). Given a
feature vector $\v{z} = f(I)$ for some image $I$, $\v{w}_c^T \v{z}$ is
the prediction score for class $c$ (the bias term is omitted here for
simplicity), and the model's class prediction for $I$ is given by
$\argmax_c \v{w}_c^T f(I)$.

@fig:pipeline shows the overview of our method. At a high level,
AlphaNet can be seen as moving the classifiers for rare classes based on
their position relative to visually similar classes. Importantly, it
updates classifiers without making any changes to the representation
space, or to other classifiers in the model. It performs a post-hoc
correction, and as such, is applicable to use cases where existing base
classifiers are either unavailable or fixed (e.g., due to commercial
interests or data privacy protections). The simplicity of our method
lends to computation advantages--AlphaNet can be trained rapidly, and
without need for multiple \acp{GPU}.
