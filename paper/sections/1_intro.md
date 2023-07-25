<!-- cSpell:ignore xwang -->

# Introduction {#sec:intro}

The significance of long-tailed distributions in real-world applications
(such as autonomous driving[@2022.Anguelov.Jiang], and medical image
analysis[@2022.Bian.Yang]) has spurred a variety of approaches for
long-tail classification[@2022.Guo.Yang]. Learning in this setting is
challenging because many classes are "rare" -- having only a small
number of training samples. Some methods re-sample more data for rare
classes in an effort to address data imbalances[@2019.Belongie.Cui;
@2019.Ma.Cao], while other methods adjust learned classifiers to
re-weight them in favor of rare classes[@2020.Kalantidis.Kang]. Both
re-sampling and re-weighting methods provide strong baselines for
long-tail classification tasks. However, state-of-the-art results are
achieved by more complex methods that, for example, learn multiple
experts[@2021.Yu.Wang; @2021.Hwang.Cai], perform multi-stage
distillation[@2021.Wu.Li], or use a combination of weight decay, loss
balancing, and norm thresholding[@2022.Kong.Alshammari].

Despite these advances, accuracy on rare classes continues to be
significantly lower than overall accuracy. For example, on ImageNet‑LT
-- a long-tailed dataset sampled from ImageNet[@2009.Fei-Fei.Deng] --
the 6-expert ensemble RIDE model[@2021.Yu.Wang] has an average accuracy
of 68.9% on frequent classes, but an average accuracy of 36.5% on rare
classes.[^note:ride_results] In addition to reducing overall accuracy,
such performance imbalances raise ethical concerns in contexts where
unequal accuracy leads to biased outcomes, such as medical
imaging[@2022.Ferrante.Lara], or face detection[@2018.Gebru.Buolamwini].
For instance, models trained on chest X‑ray images consistently
under-diagnosed minority groups[@2021.Ghassemi.Seyyed], and similarly,
cardiac image segmentation showed significant differences between racial
groups[@2021.King.Puyol].

<div id="fig:analysis">

![Predictions from cRT model on test samples from 'few' split of
ImageNet‑LT. For a misclassified sample, if the predicted class is one
of the 5 'base' split nearest neighbors (NNs) of the true class, it is
considered to be incorrectly classified as a NN. A large number of
samples are misclassified in this
way.](figures/pred_counts_imagenetlt_crt_baseline){#fig:analysis:bins}

![Sample images from two classes in ImageNet‑LT. 'Lhasa' is a 'few'
split class, and 'Tibetan terrier' is a 'base' split class. The classes
are visually very similar, leading to
misclassifications.](figures/doggies){#fig:analysis:egs}

![Per-class test accuracy of cRT model on 'few' split of ImageNet‑LT,
versus the mean Euclidean distance to 5 nearest neighbor (NN) 'base'
split classes. The line is a bootstrapped linear regression fit, and $r$
(top right) is Pearson correlation. There is a high correlation, i.e.,
'few' split classes with close 'base' split NNs are more likely to be
misclassified.](figures/cls_acc_vs_nndist_imagenetlt_crt_baseline){#fig:analysis:acc_vs_dist}

Analysis of 'few' split predictions on ImageNet‑LT.

</div>

To understand the poor rare class performance of long-tail models, we
analyzed predictions of the cRT model[@2020.Kalantidis.Kang] on test
samples from ImageNet‑LT's 'few' split (i.e., classes with limited
training samples). @fig:analysis:bins shows predictions binned into
three groups: (1) samples classified correctly; (2) samples incorrectly
classified as a visually similar 'base' split[^note:base_split] class
(e.g., 'husky' instead of 'malamute'); and (3) samples incorrectly
classified as a visually dissimilar class (e.g., 'goldfish' instead of
'malamute'). A significant portion of the misclassifications (about 23%)
are to visually similar frequent classes. @fig:analysis:egs highlights
the reason behind this issue, with samples from one pair of visually
similar classes; the differences are subtle, and can be hard even for
humans to identify. To get a quantitative understanding, we analyzed the
relationship between per-class test accuracy and mean distance of a
class to its nearest neighbors (see @sec:method for details).
@fig:analysis:acc_vs_dist shows a strong positive correlation between
accuracy and mean distance, meaning that rare classes with close
neighbors have lower test accuracy than classes with distant neighbors.

Based on these analyses, we designed a method to directly improve the
accuracy on rare classes in long-tail classification. Our method,
AlphaNet, uses information from visually similar frequent classes to
improve classifiers for rare classes. @fig:alphanet illustrates the
pipeline of our method. At a high level, AlphaNet can be seen as moving
the classifiers for rare classes based on their position relative to
visually similar classes. Importantly, AlphaNet updates classifiers
without making any changes to the representation space, or to other
classifiers in the model. It performs a post hoc correction, and as
such, is applicable to use cases where existing base classifiers are
either unavailable or fixed (e.g., due to commercial interests or data
privacy protections). The simplicity of our method lends to
computational advantages -- AlphaNet can be trained rapidly, and on top
of any classification model. We will demonstrate that AlphaNet, applied
to a variety of long-tail classification models, significantly improves
rare class accuracy on multiple datasets.

[^note:ride_results]: Results for the 6-expert model are presented in
    the GitHub repository for the original paper at
    [`github.com/frank-xwang/RIDE-LongTailRecognition/blob/main/MODEL_ZOO.md`](https://github.com/frank-xwang/RIDE-LongTailRecognition/blob/main/MODEL_ZOO.md).

[^note:base_split]: The 'base' split is the complement of the 'few'
    split, composed of classes with many training samples.
