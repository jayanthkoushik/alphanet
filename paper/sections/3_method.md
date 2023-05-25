# Methods {#sec:method}

![Pipeline for AlphaNet. Given examples from rare classes, we identify the
nearest neighbors (visually similar) examples from frequent classes and then
update each rare classifier using learned $\alpha$'s for each nearest neighbor.
The result is an improved classifier for each rare
class.](figures/pipeline){#fig:alphanet width=7.25in}

## Notation {#sec:method:notation}

In this work, we will define the distance between two classes as the distance
between their average training set representation. Given a classification
model, let $f$ be the function mapping images to vectors in $\R^d$ (typically,
this is the output of the penultimate layer in convolutional networks). For a
class $c$ with training samples $I^c_1, \dots, I^c_{n^c}$, let $z^c \equiv
(1/n^c) \sumnl_i f(I^c_i)$ \aarti{be the average training set representation}.
Given a distance metric $\mu: \R^d \times \R^d \to \R$, for classes $c_1$ and
$c_2$, we define $m_\mu(c_1, c_2) \equiv \mu(x^{c_1}, x^{c_2})$. \aarti{Do we
want to call $z^c$ as $x^c$ instead to be consistent?}

Given a long-tailed dataset, the 'few' split, $C^F$, is defined as the set of
rare classes (which form the "tail" of the training data distribution). The set
of remaining classes forms the 'base' split, $C^B$. AlphaNet is applied to
update the 'few' split classifiers using nearest neighbors from the 'base'
split. For a 'few' split class $c$, let the $k$ nearest 'base' split neighbors
(based on $m_\mu$) be $q^c_1, \dots, q^c_k$. \aarti{Some comment on how to
choose the split in practice?}

We will use the term 'classifier' to denote the linear mapping from feature
vectors to class scores. In convolutional networks, the last layer is generally
a matrix of all individual classifiers. For a class $c$, let $w^c$ be its
classifier, and let $v^c_1, \dots, v^c_k$ be the classifiers for its nearest
neighbors. Finally, let $\c{v}^c \in \R^{kd}$ \aarti{$d$ not defined,
also there are too many terms $q^c, z^c, x^c, v^c$ - maybe only introduce the
ones you need later} be a vector with the nearest neighbor classifiers
concatenated together.

## AlphaNet implementation  {#sec:method:impl}

@fig:alphanet shows the architecture of our method. AlphaNet is a small fully
connected network which maps $\c{v}^c$ to a set of coefficients $\alpha^c_1,
\dots, \alpha^c_k$. The $\alpha$ coefficients (denoted together as a vector
$\alpha^c$), are then scaled to unit 1-norm (the reasoning behind this is
explained below):
$$
  \tilde{\alpha}^c
= \alpha^c / \abs{\alpha^c}_1.
$$ {#eq:alpha_scaling}
The scaled coefficients are used to update classifiers through a linear
combination:
$$
       \shat{w}^c
\equiv w^c + \suml_{i=1}^k \tilde{\alpha}^c_i v^c_i
$$ {#eq:alphanet_update}
Due to the 1-norm scaling, we have
$$
\begin{aligned}
       \norm{\shat{w}^c - w^c}_2
&\le   \suml_{i=1}^k \abs{\tilde{\alpha}^c_i}\norm{v^c_i}_2
 \quad \text{(Cauchy-Schwarz inequality)} \\
%
&\le \max_{i=1,\dots,k} \norm{v^c_i}_2 \suml_{i=1}^k \abs{\tilde{\alpha}^c_i} \\
&=   \max_{i=1,\dots,k} \norm{v^c_i}_2 \norm{\tilde{\alpha}^c}_1 \\
&=   \max_{i=1,\dots,k} \norm{v^c_i}_2,
\end{aligned}
$$ {#eq:wdelta_norm_bound}
that is, a classifier's change is bound by the norm of its class's nearest
neighbors. Thanks to this, we do not need to update or rescale 'base' split
classifiers, which may not be possible in certain domains.

A single network is used to generate coefficients for every 'few' split class.
So, once trained, AlphaNet can be applied even to previously unseen classes.
\aarti{clarify unseen but known vs. open set - maybe add latter to future
work}

## Training {#sec:method:training}

To train AlphaNet, we have to learn parameters $\theta$ for the network which
maps $\c{v}^c$ to $\alpha^c$. We also learn a new set of bias values for the
'few' split classes, $\tilde{b}_1, \dots, \tilde{b}_{\abs{C^F}}$. We train
AlphaNet using samples from both the 'few' and 'base' splits to prevent
over-fitting. So, for a sample $(I, y)$, the prediction score is given by
$$
s(I, y; \theta, \tilde{b}) = \begin{cases}
    f(I)^T \shat{w}^y + \tilde{b}_y & y \in C^F \\
    f(I)^T w^y + b_y                & y \in C^B
\end{cases}
$$ {#eq:training_probs}
This score is used to compute the softmax cross-entropy loss on a set of
training samples, and minimized with respect to $\theta$ and $\tilde{b}$ using
\ac{SGD}.
