# Method {#sec:method}

In this work, we define the distance between two classes as the distance
between their average training set representation. Given a
classification model, let $f$ be the function mapping images to vectors
in a $d$-dimensional space (typically, this is the output of the
penultimate layer in convolutional networks). For a class $c$ with $n^c$
training samples $I^c_1, \dots, I^c_{n^c}$, let $z^c \equiv (1/n^c)
\sumnl_i f(I^c_i)$ be the average training set representation. Given a
distance function $\mu: \R^d \times \R^d \to \R$, for classes $c_1$ and
$c_2$, we define the distance between two classes as $m_\mu(c_1, c_2)
\equiv \mu(z^{c_1}, z^{c_2})$.

Given a long-tailed dataset, the 'few' split, $C^F$, is defined as the
set of classes with fewer than $T$ training samples, for some constant
$T$ (for the datasets used in our experiments, $T=20$). The set of
remaining classes forms the 'base' split, $C^B$. AlphaNet is applied to
update the 'few' split classifiers using nearest neighbors from the
'base' split. We will use the term 'classifier' to denote the linear
mapping from feature vectors to class scores. In convolutional networks,
the last layer is generally a matrix of all individual classifiers. The
bias terms are not updated by AlphaNet, and are used as is.

## AlphaNet implementation {#sec:method:impl}

![Pipeline for AlphaNet. Given a rare class, we identify the nearest
neighbor frequent classes based on visual similarity, and then update
the rare class' classifier using learned coefficients. One coefficient,
$\alpha$, is learned for each nearest neighbor. The result is an
improved classifier for the rare class.](figures/pipeline){#fig:alphanet
width=7.25in}

@fig:alphanet shows the pipeline of our method. Given a 'few' split
class $c$ with classifier $w^c$, we find its $k$ nearest 'base' split
neighbors based on $m_\mu$. Let these neighbors have classifiers $v^c_1,
\dots, v^c_k$, which are concatenated together into a vector $c{v}^c$.
AlphaNet maps $c{v}^c$ to a set of coefficients $\alpha^c_1, \dots,
\alpha^c_k$. The $\alpha$ coefficients (denoted together as a vector
$\alpha^c$), are then scaled to unit 1-norm (the reasoning behind this
will be explained later), to obtain $\tilde{\alpha}^c$:
$$
  \tilde{\alpha}^c
= \alpha^c / \norm{\alpha^c}_1.
$$ {#eq:alpha_scaling}
The scaled coefficients are used to update the 'few' split classifier
($w^c \to \shat{w}^c$) through a linear combination:
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
that is, a classifier's change is bound by the norm of its class'
nearest neighbors. Thanks to this, we do not need to update or rescale
'base' split classifiers, which may not be possible in certain domains.

A single network is used to generate coefficients for every 'few' split
class. So, once trained, AlphaNet can be applied even to classes not
seen during training. This will be further explored in future work.

## Training {#sec:method:training}

The trainable component of AlphaNet is a network (with parameters
$\theta$) which maps $\c{v}^c$ to $\alpha^c$. We use the original
classifier biases, $b$. So, given a training image $I$, the per-class
prediction scores are given by
$$
s(c; I) = \begin{cases}
       f(I)^T \shat{w}^c + b_c & c \in C^F. \\
       f(I)^T w^c + b_c        & c \in C^B.
\end{cases}
$$ {#eq:pred_scores}
These scores are used to compute the softmax cross-entropy
loss,[^note:ce] which is minimized with respect to $\theta$ using a
gradient based optimizer.

[^note:ce]: We use softmax cross-entropy loss in our experiments, but
    any loss function can be used.
