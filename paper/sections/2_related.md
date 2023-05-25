# Related work {#sec:relwork}

Combining, creating, modifying, and learning model weights are concepts that
have been implemented in many earlier models. As we review below, these
concepts appear frequently in \aarti{ensemble methods (boosting, bagging,
etc.) - where do these belong below? I suspect classifier composition, so we
should mention there? And boosting does learn the compositional weights}
transfer learning, meta-learning, zero-shot/low-shot learning, and long-tail
learning.

## Classifier creation {#sec:relwork:creation}

The process of creating new classifiers is captured within meta-learning
concepts such as learning-to-learn, transfer learning, and multi-task learning
learning[@1998.Thrun.Thrun; @1997.Wiering.Schmidhuber; @2009.Yang.Pandzsm;
@1997.Caruana.Caruana; @2016.Lillicrap.Santoro]. These approaches generalize to
novel tasks by learning shared information from a set of related tasks. Many
studies find that shared information is embedded within model weights, and,
thus, aim to learn structure within learned models to directly modify the
weights of a different network[@1993.Schmidhuber.Schmidhuber;
@1992.Schmidhuber.Schmidhuber; @2016.Vedaldi.Bertinetto; @2016.Le.Ha;
@2018.Levine.Finn; @2017.Vedaldi.Rebuffi; @2017.Krishnamurthy.Sinha;
@2017.Yu.Munkhdalai]. Other studies go even further and instead of modifying
networks, they create entirely new networks exclusively from training
samples[@2013.Ng.Socher; @2015.Salakhutdinov.Ba; @2016.Han.Noh]. In contrast,
AlphaNet only combines existing classifiers, without having to create new
classifiers or train networks from scratch.

## Classifier or feature composition {#sec:relwork:composition}

In various classical approaches, there has been work that learns better
embedding spaces for image annotation[@2010.Usunier.Weston], or uses
classification scores as useful features[@2012.Forsyth.Wang]. However, these
approaches do not attempt to compose classifiers nor do they address the
long-tail problem. Within non-deep methods in classic transfer learning, there
have been attempts to use and combine \acp{SVM}. In one method[@2005.Singer.Tsochantaridis],
\acp{SVM} are trained per object instance, and a hierarchical structure is
required for combination in the datasets of interest. Such a structure is
typically not guaranteed nor provided in long-tailed datasets. Another \ac{SVM}
method uses regularized minimization to learn the coefficients necessary to
combine patches from other classifiers[@2012.Zisserman.Aytar].

While these approaches are conceptually similar to our method, AlphaNet has the
additional advantage of _learning_ the compositional coefficients without any
hyper-parameters and tuning. Specifically, different novel classes will have
their own sets of composition coefficients, and similar novel classes will
naturally have similar coefficients. Learning such varying sets of coefficients
is difficult in previous classical approaches, which either learn a fixed set
of alphas \aarti{alphas $\rightarrow$ coefficients} for all novel classes or
are forced to introduce more complex group sparsity-like constraints
\aarti{ref?}. Finally, in zero-shot learning there exist methods which compose
classifiers of known visual concepts to learn a completely new
classifier[@2013.Elgammal.Elhoseiny; @2017.Hebert.Misra; @2015.Salakhutdinov.Ba;
@2016.Sha.Changpinyo]. However, such composition is often guided by additional
attribute supervision or textual description, which are not needed by AlphaNet.

## Learning transformations between models and classes {#sec:relwork:transformation}

Other studies have demonstrated different ways of learning transformations to
modify model weights in an attempt to learn these transformations with \ac{SGD}
optimization[@2016.Freitas.Andrychowicz; @2017.Larochelle.Ravi]. Additionally,
there is empirical evidence[@2016.Hebert.Wang] showing the existence of a
generic nonlinear transformation from small-sample to large-sample models for
different types of feature spaces and classifier models. Finally, in the case
where one learns the transformation from the source function to a related
target function, there are theoretical guarantees on performance[@2016.Poczos.Du].
\aarti{update ref to NIPS17}AlphaNet is similar in that we likewise infer that
our target classifier is a transformation from a set of source classifiers.

## Zero-shot/low-shot learning {#sec:relwork:lowshot}

Meta-learning, transfer learning, and learning-to-learn are frequently applied
to the domain of low-shot learning[@2006.Perona.Fei-Fei;
@2015.Salakhutdinov.Koch; @2015.Tenenbaum.Lake; @2016.Lillicrap.Santoro;
@2016.Hebert.Wang; @2016.Hoiem.Li; @2017.Girshick.Hariharan;
@1993.Shah.Bromley; @2017.Zemel.Snell; @2017.Phoenix.George;
@2017.Hebert.Wangpln; @2015.Schmid.Akata]. A wide variety of prior studies have
attempted to transfer knowledge from tasks with abundant data to completely
novel tasks[@2016.Wierstra.Vinyals; @2017.Larochelle.Ravi;
@2016.Vedaldi.Bertinetto]. However, the explicit nature of low-shot learning
consisting of tasks with small fixed samples means that these approaches do not
generalize well beyond the arbitrary few tasks. This is a significant problem
as the visual world clearly involves a wide set of tasks with continuously
varying amounts of information.

## Long-tail learning {#sec:relwork:longtail}

The restrictions of low-shot learning have directly led to the new \aarti{not
sure I'd call this new - maybe new paradigm in computer vision?} paradigm
referred to as long-tail learning, where data samples are continuously
decreasing \aarti{with increasing number of classes} and the data distribution
closely models that of the visual world. Recent work achieves state-of-the-art
performance on long-tailed recognition by learning multiple
experts[@2021.Hwang.Cai; @2020.Yu.Wang]. Both of these complex ensemble methods
require a two-stage training method. A somewhat different approach re-balances
the samples at different stages of model training[@2019.Ma.Cao], attempts to
transfer features from common classes to rare classes[@2019.Yu.Liu], or
transfers intra-class variance[@2019.Chandraker.Yin]. However, approaches to
knowledge transfer require complex architectures, such as a specialized
attention mechanism and memory models[@2019.Yu.Liu]. While most studies have
largely focused on representation space transferability or complex ensembles,
recent work establishes a strong baseline by exploring the potential of
operating in classifier space[@2019.Kalantidis.Kang]. Results suggest that
decoupling model representation learning and classifier learning is a more
efficient way to approach long-tailed learning. Specifically, methods
normalizing classifiers and adjusting classifiers only using re-sampling
strategies achieve good performance \aarti{ref - other than just one [21]?}.
Such successes in working only with classifiers support our general concept
that combining strong classifiers in AlphaNet is a natural and direct way to
improve upon weak classifiers.
