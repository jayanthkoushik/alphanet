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

To understand the poor rare class performance of long-tail models, we
analyzed the predictions of the \ac{RIDE} model[@ride] on 'few' split test
samples in ImageNet-LT (see @sec:method for details).
@fig:analysis:bins shows predictions on a test set binned into three
groups: 1) samples predicted correctly; 2) samples incorrectly predicted
as a visually similar class (e.g., predicting 'husky' instead of
'malamute'); and 3) samples incorrectly predicted as a visually
dissimilar class (e.g., predicting 'car (anything not animal)' instead
of 'malamute'). A significant portion of the misclassifications (about
xx%) are to visually similar classes. @fig:analysis:egs shows samples
from one pair of visually similar classes; the differences are subtle,
and can be hard even for humans to identify. We next analyzed the
relationship between test accuracy and mean nearest neighbor distance
(i.e., the average $m_\mu$ between a class and its nearest neighbors).
@fig:analysis:acc_vs_dist shows a strong positive correlation between
accuracy and mean distance --'few' split classes with close neighbors
have lower test accuracy than classes with distant neighbors.

Based on these analyses, we designed a method, AlphaNet, to improve
classifiers for rare classes using information from visually similar
*frequent* classes.
@fig:pipeline shows the overview of our method. At a high level,
AlphaNet can be seen as moving the classifiers for rare classes based on
their position relative to visually similar classes. Importantly,
AlphaNet updates classifiers without making any changes to the
representation space, or to other classifiers in the model. It performs
a post-hoc correction, and as such, is applicable to use cases where
existing base classifiers are either unavailable or fixed (e.g., due to
commercial interests or data privacy protections). The simplicity of our
method lends to computation advantages--AlphaNet can be trained rapidly,
and without need for multiple \acp{GPUs}.
