# Introduction {#sec:intro}

The significance of long-tailed distributions in real-world applications
(such as autonomous driving[@2018.Malla.Narayanan] and medical image
analysis[@2022.Bian.Yang]) has spurred a variety of approaches for
long-tail classification.[@2022.Guo.Yang] Learning in this setting is
challenging because many classes are "rare" -- having only a small
number of training samples. Some methods re-sample more data for rare
classes in an effort to address data imbalances,[@2019.Belongie.Cui;
@2019.Ma.Cao] while other methods adjust learned classifiers to
re-weight them in favor of rare classes.[@2019.Kalantidis.Kang] Both
re-sampling and re-weighting methods provide strong baselines for
long-tail classification tasks. However, state-of-the-art results are
achieved by more complex methods that, for example, learn multiple
experts,[@2020.Yu.Wang; @2021.Hwang.Cai] perform multi-stage
distillation,[@2021.Wu.Li] or use a combination of weight balancing,
data re-sampling, and loss decay.[@2022.Kong.Alshammari]

Despite these advances, accuracy on rare classes continues to be
significantly worse than overall accuracy using these methods. For
example, on the ImageNet‑LT dataset, the 6-expert ensemble RIDE
model[@2020.Yu.Wang] has an average accuracy of 68.9% on frequent
classes, but an average accuracy of 36.5% on rare classes. In addition
to reducing overall accuracy, such performance imbalances raise ethical
concerns in contexts where unequal accuracy leads to biased outcomes,
for instance in medical imaging[@2022.Ferrante.Lara] or face
detection.[@buolamwini2018gender] For example, models trained on chest
X-rays consistently under-diagnosed minority
groups.[@2021.Ghassemi.Seyyed] Similarly, cardiac image segmentation
showed significant differences between racial groups.[@2021.King.Puyol]

<div id="fig:analysis">

![Predictions from cRT model on test samples from 'few' split of
ImageNet‑LT. A large percentage of samples are misclassified as a
visually similar 'base' split
class.](figures/pred_counts_imagenetlt_crt_baseline){#fig:analysis:bins}

![Sample images from two classes in ImageNet‑LT. 'Lhasa' is a 'few'
split class, and 'Tibetan terrier' is a 'base' split class. The classes
are visually very similar, leading to
misclassifications.](figures/doggies){#fig:analysis:egs}

![Per-class test accuracy of cRT model on 'few' split of ImageNet‑LT,
versus the mean distance to 5 nearest neighbors from the 'base' split.
The line is a bootstrapped linear regression fit, and $r$ (top right) is
Pearson correlation. There is a high correlation, i.e., 'few' split
classes with close nearest neighbors were more likely to be
misclassified.](figures/cls_acc_vs_nndist_imagenetlt_crt_baseline){#fig:analysis:acc_vs_dist}

Analysis of 'few' split predictions on ImageNet‑LT.

</div>

To understand the poor rare class performance of long-tail models, we
analyzed predictions of the cRT model[@2019.Kalantidis.Kang] on test
samples from 'few' split classes (i.e. classes with limited training
samples) in ImageNet‑LT[@2019.Yu.Liu] -- a long-tailed dataset based on
ImageNet. @fig:analysis:bins shows predictions binned into three groups:
(1) samples predicted correctly; (2) samples incorrectly predicted as a
visually similar class (e.g., predicting 'husky' instead of 'malamute');
and (3) samples incorrectly predicted as a visually dissimilar class
(e.g., predicting 'car' instead of 'malamute'). A significant portion of
the misclassifications (about 26%) are to visually similar classes.
@fig:analysis:egs shows samples from one pair of visually similar
classes; the differences are subtle, and can be hard even for humans to
identify. We also analyzed the relationship between per-class test
accuracy and mean distance of a class to its nearest neighbors (see
@sec:method for details). @fig:analysis:acc_vs_dist shows a strong
positive correlation between accuracy and mean distance, that is, 'few'
split classes with close neighbors have lower test accuracy than classes
with distant neighbors.

Based on these analyses, we designed a method to directly improve the
accuracy of rare classes in long-tail classification. Our method,
AlphaNet, uses information from visually similar frequent classes to
improve classifiers for rare classes. @fig:alphanet illustrates the
pipeline of our method. At a high level, AlphaNet can be seen as moving
the classifiers for rare classes based on their position relative to
visually similar classes. Importantly, AlphaNet updates classifiers
without making any changes to the representation space, or to other
classifiers in the model. It performs a _post-hoc_ correction, and as
such, is applicable to use cases where existing base classifiers are
either unavailable or fixed (e.g., due to commercial interests or data
privacy protections). The simplicity of our method lends to
computational advantages -- AlphaNet can be trained rapidly, and on top
of any classification model. We will demonstrate that AlphaNet
significantly improves the accuracy of rare classes for a variety of
long-tail classification models across multiple datasets.
