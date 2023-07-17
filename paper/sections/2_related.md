<!-- cSpell:ignore Torralba -->

# Related work {#sec:relwork}

Combining, creating, modifying, and learning model weights are concepts
that have been implemented in many earlier models. As we review below,
these concepts appear frequently in transfer learning, meta-learning,
zero-shot/low-shot learning, and long-tail learning.

## Classifier creation {#sec:relwork:creation}

The process of creating new classifiers is captured within meta-learning
concepts such as learning-to-learn, transfer learning, and multi-task
learning.[@1998.Thrun.Thrun; @1997.Wiering.Schmidhuber;
@2009.Yang.Pandzsm; @1997.Caruana.Caruana; @2016.Lillicrap.Santoro]
These approaches generalize to novel tasks by learning shared
information from a set of related tasks. Many studies find that shared
information is embedded within model weights, and, thus aim to learn
structure within learned models to directly modify the weights of a
different network.[@1993.Schmidhuber.Schmidhuber;
@1992.Schmidhuber.Schmidhuber; @2016.Vedaldi.Bertinetto; @2016.Le.Ha;
@2018.Levine.Finn; @2017.Vedaldi.Rebuffi; @2017.Krishnamurthy.Sinha;
@2017.Yu.Munkhdalai] Other studies go even further and instead of
modifying networks, they create entirely new networks exclusively from
training samples.[@2013.Ng.Socher; @2015.Salakhutdinov.Ba;
@2016.Han.Noh] In contrast, AlphaNet only combines existing classifiers,
without having to create new classifiers or train networks from scratch.

## Classifier or feature composition {#sec:relwork:composition}

There have been works that learn better embedding spaces for image
annotation,[@2010.Usunier.Weston] or use classification scores as useful
features.[@2012.Forsyth.Wang] However, these approaches do not attempt
to compose classifiers nor do they address the long-tail problem. For
transfer learning with non-deep methods, there have been attempts to use
and combine support vector machines (SVMs). In one
method,[@2005.Singer.Tsochantaridis] SVMs are trained per object
instance, and a hierarchical structure is required for combination in
the datasets of interest. However, such a structure is typically neither
guaranteed nor provided in long-tailed datasets. Another SVM method uses
regularized minimization to learn the coefficients necessary to combine
patches from other classifiers.[@2012.Zisserman.Aytar]

While these approaches are conceptually similar to our method, AlphaNet
has the additional advantage of _learning_ the compositional
coefficients. Specifically, different novel classes will have their own
set of coefficients, and similar novel classes will naturally have
similar coefficients. Finally, in zero-shot learning there exist methods
which compose classifiers of known visual concepts to learn a completely
new classifier.[@2013.Elgammal.Elhoseiny; @2017.Hebert.Misra;
@2015.Salakhutdinov.Ba; @2016.Sha.Changpinyo] However, such composition
is often guided by supervision from additional attributes or textual
descriptions, which are not needed by AlphaNet.

### Boosting {#sec:relwork:composition:boosting}

The idea of composing weak classifiers to build strong classifiers bears
resemblance to the idea of boosting.[@schapire1990strength] The popular
AdaBoost[@freund1995desicion] linearly combines classifiers based on a
single feature (e.g., decision stumps), and iteratively re-weights
training samples based on their error. For the case of multi-class
classification, Torralba et\ al. (2007)[@torralba2007sharing] build a
classifier that combines several binary classifiers, each designed to
separate a single class from the others. Their method identifies common
features that be shared across classifiers, which reduces the
computational load, and the amount of training data required.

It is important to note that boosting methods employ a different form of
composition that our methods. Specifically, our focus is on
classification methods where the _performance on a subset of classes_ is
poor. Unlike boosting methods, we do not incorporate additional features
-- improvements are made by adjusting classifiers within the learned
representation space.

## Learning transformations between models and classes {#sec:relwork:transformation}

Some studies have attempted to learn transformations of model weights
with gradient based optimization.[@2016.Freitas.Andrychowicz;
@2017.Larochelle.Ravi] Additionally, there is empirical
evidence[@2016.Hebert.Wang] showing the existence of a generic nonlinear
transformation from small-sample to large-sample models for different
types of feature spaces and classifier models. Finally, in the case
where one learns the transformation from the source function to a
related target function, there are theoretical guarantees on
performance.[@2016.Poczos.Du] AlphaNet is similar in that we likewise
infer that our target classifier is a transformation from a set of
source classifiers.

## Zero-shot/low-shot learning {#sec:relwork:lowshot}

Meta-learning, transfer learning, and learning-to-learn are frequently
applied to the domain of low-shot learning.[@2006.Perona.Fei-Fei;
@2015.Salakhutdinov.Koch; @2015.Tenenbaum.Lake; @2016.Lillicrap.Santoro;
@2016.Hebert.Wang; @2016.Hoiem.Li; @2017.Girshick.Hariharan;
@1993.Shah.Bromley; @2017.Zemel.Snell; @2017.Phoenix.George;
@2017.Hebert.Wangpln; @2015.Schmid.Akata] A wide variety of prior
studies have attempted to transfer knowledge from tasks with abundant
data to completely novel tasks.[@2016.Wierstra.Vinyals;
@2017.Larochelle.Ravi; @2016.Vedaldi.Bertinetto] However, due to the
nature of low-shot learning, these approaches are limited to a small
number of tasks, which is problematic since the visual world involves a
large number of tasks with varying amounts of information.

## Long-tail learning {#sec:relwork:longtail}

The restrictions of low-shot learning have been addressed by the
paradigm referred to as long-tail learning, where the distribution of
class sizes (number of training samples) closely models that of the
visual world; many classes have only a few samples, while a few classes
have many samples. Recent work achieves state-of-the-art performance on
long-tailed recognition by learning multiple experts.[@2021.Hwang.Cai;
@2020.Yu.Wang] Both of these complex ensemble methods require a
two-stage training method. Other approaches re-balance the class sizes
at different stages of model training,[@2019.Ma.Cao] transfer features
from common classes to rare classes,[@2019.Yu.Liu] or transfer
intra-class variance.[@2019.Chandraker.Yin] However, approaches for
knowledge transfer require complex architectures, such as a specialized
attention mechanism with memory.[@2019.Yu.Liu] While recent studies have
largely focused on representation space transferability or complex
ensembles, strong baselines have been established by exploring the
potential of operating in classifier space.[@2019.Kalantidis.Kang]
Results suggest that decoupling model representation learning and
classifier learning is a more efficient way to approach long-tailed
learning. Specifically, methods normalizing classifiers and adjusting
classifiers using only re-sampling strategies achieve good
performance.[@2019.Kalantidis.Kang] These strong baselines support our
approach of operating in classifier space -- AlphaNet combines strong
classifiers to improve weak classifiers.
