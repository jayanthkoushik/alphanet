<!-- cSpell:ignore Kang, matplotlib, seaborn -->

# Implementation details {#sec:impl}

Experiments were run using the PyTorch[@2019.Chintala.Paszke] library.
We used the container implementation provided by NVIDIA GPU cloud
(NGC).[^note:container_link] Code to reproduce experimental results is
available on GitHub.[^note:self_repo]

## Datasets {#sec:impl:datasets}

{% include tables/appendix/static/dataset_stats.md %}

Details about the long-tailed datasets used in our experiments are shown
in @tbl:dataset_stats. For ImageNet‑LT and Places‑LT, we used splits
from Kang et\ al.[@2020.Kalantidis.Kang], available on
GitHub.[^note:cls_bal_repo] For CIFAR‑100‑LT, we used the implementation
of Wang et\ al.[@2021.Yu.Wang], with imbalance factor 100, also
available on GitHub.[^note:ride_repo]

### Splits {#sec:impl:datasets:splits}

For all datasets, the 'many', 'medium', and 'few' splits are defined
using the same limits on per-class training samples: less than 20 for
the 'few' split, between 20 and 100 for the 'medium' split, and more
than 100 for the 'many' split. The actual minimum and maximum per-class
training samples for each split are shown in @tbl:dataset_splits.

## Baseline models {#sec:impl:baselines}

Baseline model architectures are shown in @tbl:baseline_archs. All
models used backbones made of residual networks --
ResNets[@2016.Sun.He], and ResNeXts[@2017.He.Xie]. Whenever we refer to
a model, the architecture corresponding to the dataset is used. For
example, cRT used the ResNeXt‑50 architecture on ImageNet‑LT, and the
ResNet‑152 architecture on Places‑LT.

For all models except LTR, we used model weights provided by the
respective authors. For LTR, we retrained the model using code provided
by the authors,[^note:ltr_repo] with some modifications: (1) for
consistency, we used the same CIFAR‑100‑LT data splits used for training
the RIDE model, and (2) we performed second stage training --
fine-tuning with weight decay and norm thresholding -- for a fixed 10
epochs.

{% include tables/appendix/static/dataset_splits.md %}

### Feature and classifier extraction {#sec:impl:baselines:extract}

{% include tables/appendix/static/baseline_archs.md %}

In most cases, we simply used the output of a model's penultimate layer
as features, and used the weights (including bias) of the last layer as
the classifier. Exceptions are listed below:

* LWS: We multiplied classifier weights with the learned scales.

* RIDE: We used the 6-expert teacher model, and saved classifiers from
  each expert after normalizing and scaling as in the model. For
  AlphaNet training, we created a single classifier by concatenating the
  expert classifier weights and biases. Similarly, features extracted
  from individual experts were concatenated after normalizing. During
  prediction, the individual experts and features were re-extracted, and
  experts were applied to their corresponding features to get 6
  predictions, which were then averaged. So AlphaNet learned
  coefficients to update all 6 experts simultaneously.

* LTR: We used the model fine-tuned with weight decay and norm
  thresholding. This creates a classifier with small norm, and
  correspondingly small prediction scores. So, during AlphaNet training,
  we multiplied all prediction scores by 100, which is equivalent to
  setting the softmax temperature to 0.01.

## Training {#sec:impl:training}

For the main experiments, 5 nearest neighbors were selected for each
'few' split class, based on Euclidean distance. Hyper-parameter settings
used during training are shown in @tbl:hyperparams. ImageNet‑LT and
Places‑LT have a validation set, which was used to select the best
model. This was controlled by the 'minimum epochs' parameter. After
training for at least this many epochs, model weights were saved at the
end of each epoch. Finally, the best model was selected based on overall
validation accuracy, and used for testing. For CIFAR‑100‑LT and
iNaturalist, we simply trained for a fixed number of epochs.

{% include tables/appendix/static/hyperparams.md %}

## Results {#sec:impl:results}

All experiments were repeated 10 times from different random
initializations, and unless specified otherwise, results are average
values. In tables, the standard deviation is shown in superscript. We
regenerated baseline results, and these match published values, except
in the case of LTR, since it was retrained, and, for consistency, we did
not use data augmentation at test time. Plots were generated with
Matplotlib[@2007.Hunter], using the Seaborn library[@2021.Waskom]. Error
bars in figures represent 95% confidence intervals, estimated using
10,000 bootstrap resamples.

<!-- cSpell: disable -->

[^note:container_link]:
    [`catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch`](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch),
    version 22.06.

[^note:self_repo]:
    [`github.com/jayanthkoushik/alphanet`](https://github.com/jayanthkoushik/alphanet).

[^note:cls_bal_repo]:
    [`github.com/facebookresearch/classifier-balancing`](https://github.com/facebookresearch/classifier-balancing).

[^note:ride_repo]:
    [`github.com/frank-xwang/RIDE-LongTailRecognition`](https://github.com/frank-xwang/RIDE-LongTailRecognition).

[^note:ltr_repo]:
    [`github.com/ShadeAlsha/LTR-weight-balancing`](https://github.com/ShadeAlsha/LTR-weight-balancing).

<!-- cSpell: enable -->
