# Method {#sec:method}

Our problem setting is multi-class classification with $C$ classes,
where each input has a corresponding class label in $\set{0, \dots, C -
1}$, and the goal is to learn a mapping from inputs to labels. We are
specifically interested in visual recognition -- inputs are images, and
classes are object categories. AlphaNet is applied over a pre-trained
classification model. We assume that this model can be decoupled into
two parts: the first part maps images to feature vectors, and the second
part maps feature vectors to "scores", one for each of the $C$ classes.
The prediction for an image is the class index with largest
corresponding score. Typically (and in all our experiments), for a
convolutional network, the feature vector for an image is the output of
the penultimate layer, and the last layer is a linear mapping. So, each
classifier is a vector, and the score for a class is the dot product of
the feature vector with the corresponding classifier. Typically a bias
term is present, which is added to the dot product. We do not modify
this term, and use it as-is if present.

In this work, we define the distance between two classes as the distance
between their average training set representation. Let $f$ be the
function mapping images to feature vectors in a $d$-dimensional
representation space. For a class $c$ with $n^c$ training samples
$I^c_1, \dots, I^c_{n^c}$, let $\v{z}^c \equiv (1/n^c) \sumnl_i
f(I^c_i)$ be the average training set representation. Given a distance
function $\mu: \R^d \times \R^d \to \R$, we define the distance between
two classes $c_1$ and $c_2$ as $m_\mu(c_1, c_2) \equiv \mu(\v{z}^{c_1},
\v{z}^{c_2})$.

Given a long-tailed dataset, the 'few' split, $C^F$, is defined as the
set of classes with fewer than $T$ training samples, for some constant
$T$ (equal to 20 for the datasets used in this work). The remaining
classes form the 'base' split, $C^B$. AlphaNet is applied to update the
'few' split classifiers using nearest neighbors from the 'base' split.

## AlphaNet implementation {#sec:method:impl}

![Pipeline for AlphaNet. Given a rare class, we identify the nearest
neighbor frequent classes based on visual similarity, and then update
the rare class' classifier using learned coefficients. One coefficient,
$\alpha$, is learned for each nearest neighbor. The result is an
improved classifier for the rare class.](figures/pipeline){#fig:alphanet
width=7.25in}

@fig:alphanet shows the pipeline of our method. Given a 'few' split
class $c$ with classifier $\v{w}^c$, we find its $k$ nearest 'base'
split neighbors based on $m_\mu$. Let these neighbors have classifiers
$\v{v}^c_1, \dots, \v{v}^c_k$, which are concatenated together into a
vector $\c{\v{v}}^c$. AlphaNet maps $\c{\v{v}}^c$ to a set of
coefficients $\alpha^c_1, \dots, \alpha^c_k$. The $\alpha$ coefficients
(denoted together as a vector $\v{\alpha}^c$), are then scaled to unit
1-norm -- the reasoning behind this will be explained later -- to obtain
$\tilde{\v{\alpha}}^c$:
$$
       \tilde{\v{\alpha}}^c
\equiv \v{\alpha}^c / \norm{\v{\alpha}^c}_1.
$$ {#eq:alpha_scaling}
The scaled coefficients are used to update the 'few' split classifier
($\v{w}^c \to \shat{\v{w}}^c$) through a linear combination:
$$
       \shat{\v{w}}^c
\equiv \v{w}^c + \suml_{i=1}^k \tilde{\alpha}^c_i \v{v}^c_i
$$ {#eq:alphanet_update}
Due to the 1-norm scaling, we have
$$
\begin{aligned}
       \norm{\shat{\v{w}}^c - \v{w}^c}_2
&\le   \suml_{i=1}^k \abs{\tilde{\alpha}^c_i}\norm{\v{v}^c_i}_2
 \quad \text{(Cauchy-Schwarz inequality)} \\
%
&\le \max_{i=1,\dots,k} \norm{\v{v}^c_i}_2 \suml_{i=1}^k \abs{\tilde{\alpha}^c_i} \\
&=   \max_{i=1,\dots,k} \norm{\v{v}^c_i}_2 \norm{\tilde{\v{\alpha}}^c}_1 \\
&=   \max_{i=1,\dots,k} \norm{v^c_i}_2,
\end{aligned}
$$ {#eq:wdelta_norm_bound}
that is, a classifier's change is bound by the norm of its class'
nearest neighbors. Thanks to this, we do not need to update or rescale
'base' split classifiers, which may not be possible in certain domains.

A single network is used to generate coefficients for every 'few' split
class. So, once trained, AlphaNet can be applied even to classes not
seen during training. This will be further explored in future work.

## Training {#sec:method:training}

The trainable component of AlphaNet is a network (with parameters
$\v{\theta}$) which maps $\c{\v{v}}^c$ to $\v{\alpha}^c$. We use the
original classifier biases, $\v{b}$ (one per class). So, given a
training image $I$, the per-class prediction scores are given by
$$
s(c; I) = \begin{cases}
       f(I)^T \shat{\v{w}}^c + b_c & c \in C^F. \\
       f(I)^T \v{w}^c + b_c        & c \in C^B.
\end{cases}
$$ {#eq:pred_scores}
That is, class scores are unchanged for 'base' split classes, and are
computed using updated classifiers for 'few' split classes. These scores
are used to compute the sample loss (softmax cross-entropy in our
experiments); $\v{\theta}$ can then be updated iteratively using a
gradient based optimizer to reduce average sample loss estimated using
mini-batches of samples.
