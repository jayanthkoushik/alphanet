# Method {#sec:method}

Given a long-tailed dataset, we will refer to the set of rare classes
(which form the "tail" of the training data distribution) as the 'few'
split, and the set of remaining classes as the 'base' split. AlphaNet is
applied to update the 'few' split classifiers using nearest neighbors
from the 'base' split.

Let $\mathcal{C}^F$ and $\mathcal{C}^B$ be the set of 'few' and 'base'
split classes respectively.  Following the definition in @sec:intro, for
a 'few' split class $c \in \mathcal{C}^F$, let the $k$ nearest 'base'
split neighbors be $q^c_1, \dots, q^c_k$. Let $\v{w}^c$ be the
classifier (a vector in $\R^d$) for $c$, and let $\v{v}^c_1, \dots,
\v{v}^c_k$ be the classifiers for its nearest neighbors. Define
$\c{\v{v}}^c \in \R^{kd}$ as the concatenated vector of $c$'s nearest
neighbor classifiers.

## AlphaNet {#sec:method:alphanet}

@fig:alphanet shows the architecture of our method. AlphaNet is a small
fully connected network which maps $\c{\v{v}}^c$ to a set of
coefficients $\alpha^c_1, \dots, \alpha^c_k$. The $\alpha$ coefficients
(denoted together as a vector $\v{\alpha}^c$), are then scaled to unit
1-norm (the reasoning behind this is explained below):
$$
  \tilde{\v{\alpha}}^c
= \v{\alpha}^c / \norm{\v{\alpha}^c}_1.
$$ {#eq:alpha_scaling}
The scaled coefficients are used to update classifiers through a linear
combination:
$$
       \shat{\v{w}}^c
\equiv \v{w}^c + \suml_{i=1}^k \tilde{\alpha}^c_i \v{v}^c_i
=      \v{w}^c + V^c \tilde{\v{\alpha}}^c,
$$ {#eq:alphanet_update}
where $V$ is a matrix of the nearest neighbor vectors. Due to the 1-norm
scaling, we have
$$
\begin{aligned}
       \norm{\shat{\v{w}}^c - \v{w}^c}_2
&\le   \suml_{i=1}^k \abs{\tilde{\alpha}^c_i}\norm{\v{v}^c_i}_2
 \quad \text{(Cauchy-Schwarz inequality)} \\
%
&\le \max_{i=1,\dots,k} \norm{\v{v}^c_i}_2 \suml_{i=1}^k \abs{\tilde{\alpha}^c_i} \\
&=    \max_{i=1,\dots,k} \norm{\v{v}^c_i}_2 \norm{\tilde{\v{\alpha}}^c}_1 \\
&=    \max_{i=1,\dots,k} \norm{\v{v}^c_i}_2,
\end{aligned}
$$ {#eq:wdelta_norm_bound}
i.e., a classifier's change is bound by the norm of its class' nearest
neighbors. Thanks to this, we do not need to update or rescale 'base'
split classifiers, which may not be possible in certain domains.

A single network is used to generate coefficients for every 'few' split
class. So, once trained, AlphaNet can be applied even to previously
unseen classes.

## Training {#sec:method:training}

To train AlphaNet, we have to learn parameters $\v{\theta}$ for the
network which maps $\c{\v{v}}^c$ to $\v{\alpha}^c$. We also learn a new
set of bias values for the 'few' split classes, $\tilde{b}_1, \dots,
\tilde{b}_{\abs{\mathcal{C}^F}}$. We train AlphaNet using samples from
both the 'few' and 'base' splits to prevent over-fitting. So, for a
sample $(I, y)$, the prediction score is given by
$$
s(I, y; \v{\theta}, \tilde{\v{b}}) = \begin{cases}
    f(I)^T \shat{\v{w}}^y + \tilde{b}_y & y \in \mathcal{C}^F \\
    f(I)^T \v{w}^y + b_y                & y \in \mathcal{C}^B
\end{cases}
$$ {#eq:training_probs}
This score is used to compute the softmax cross-entropy loss on a set of
training samples, and minimized with respect to $\v{\theta}$ and
$\tilde{\v{b}}$ using \ac{SGD}.
