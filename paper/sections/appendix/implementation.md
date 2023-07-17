<!-- cSpell:ignore Kang, matplotlib, seaborn -->

# Implementation details {#sec:impl_details}

Experiments were run using the PyTorch[@paszke2019pytorch] library. We
used the container implementation provided by NVIDIA GPU cloud
(NGC).[^note:container_link] Code to reproduce experimental results is
available on GitHub.[^note:self_repo]

## Datasets

{% include tables_stats/dataset_stats.md %}

Details about the long-tailed datasets used in our experiments are shown
in @tbl:dataset_stats. For ImageNet‑LT and Places‑LT, we used splits
from Kang et\ al. (2019),[@2019.Kalantidis.Kang] available on
GitHub.[^note:cls_bal_repo] For CIFAR‑100‑LT, we used the 100 imbalance
implementation of Wang et\ al. (2020),[@2020.Yu.Wang], also available on
GitHub.[^note:ride_repo]

### Splits

For all datasets, the 'many', 'medium', and 'few' splits are defined
using the same limits on per-class training samples: less than 20 for
the 'few' split, between 20 and 100 for the 'medium' split, and more
than 100 for the 'many' split. The actual minimum and maximum per-class
training samples for each split are shown in @tbl:dataset_splits.

## Baseline models

Baseline model architectures are shown in @tbl:baseline_archs. All
models used backbones made of residual networks -- ResNets,[@he2016deep]
and ResNeXts.[@xie2017aggregated] Whenever we refer to a model, the
architecture corresponding to the dataset is used. For example, cRT used
the ResNeXt‑50 architecture on ImageNet‑LT, and the ResNet‑152
architecture on Places‑LT.

For all models except LTR, we used model weights provided by the
respective authors. For LTR, we retrained the model using code provided
by the authors,[^note:ltr_repo] with some modifications: (1) for
consistency, we used the same CIFAR‑100‑LT data splits used for training
the RIDE model, and (2) we performed second stage training --
fine-tuning with weight decay and norm thresholding -- for a fixed 10
epochs.

{% include tables_stats/dataset_splits.md %}

### Feature and classifier extraction

{% include tables/appendix/static/baseline_archs.md %}

In most cases, we simply used the flattened output of a model's
penultimate layer as features, and used the weights (including bias) of
the last layer as the classifier. Exceptions to this are listed below:

* LWS: We multiplied classifier weights with the per-class scales.

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

## Training

For the main experiments, 5 nearest neighbors were selected for each
'few' split class, based on Euclidean distance. Hyper-parameter settings
used during training are shown in @tbl:hyperparams. All datasets except
CIFAR‑100‑LT have a validation set, which was used to select the best
model. This was controlled by the 'minimum epochs' parameter. After
training for at least this many epochs, model weights were saved at the
end of each epoch. Finally, the best model was selected based on overall
validation accuracy, and used for testing. For CIFAR‑100‑LT, we simply
trained for a fixed number of epochs.

{% include tables/appendix/static/hyperparams.md %}

## Results

All experiments were repeated 10 times from different random
initializations, and unless specified otherwise, results (e.g.,
accuracy) are average values. In tables, the standard deviation is shown
in superscript. We regenerated baseline results, and these match
published values, except in the case of LTR, which we retrained.
Additionally, for consistency, we also did not use data augmentation at
test time.

Plots were generated with Matplotlib,[@hunter2007matplotlib] using the
Seaborn library.[@waskom2021] Wherever applicable, error bars show 95%
confidence intervals, estimated using 10,000 bootstrap resamples.

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
