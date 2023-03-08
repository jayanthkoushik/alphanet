# Methods {#sec:method}

## Notation {#sec:method:notation}

Throughout this work, we will define the distance between two classes as
the distance between their average training set representation. Given a
classification model, let $f$ be the function mapping images to vectors
in $\R^d$ (typically, this is the output of the penultimate layer in
convolutional networks). For a class $c$ with training samples $I^c_1,
\dots, I^c_{n_c}$, let $x^c \equiv (1/n_c) \sumnl_i f(I^c_i)$. Given a
distance metric $\mu: \R^d \times \R^d \to \R$, for classes $c_1$ and
$c_2$, we define $m_\mu(c_1, c_2) \equiv \mu(x^{c_1}, x^{c_2})$.

We first group up our classes into two broad splits: our 'base' split
with $B$ number of classes and our 'few' split with $F$ classes. The
'base' classes contain classes with many examples, and conversely the
'few' classes contain classes with few examples. We denote by $N = B +
F$ the total number of classes.

<!-- Additionally, we denote a generic sample by $I$ and its corresponding
label by $y$. We split the training set into two subsets of pairs
$(I_i,y_i)$: the subset $X_{base}$ contains samples from the $B$
classes, and the subset $X_{few}$ contains samples from the $F$ classes. -->

<!-- Finally, we work with a fixed pre-trained model with classifiers $W_j$
and biases $b_j$, $j \in (1, \dots, N)$; here, classifiers are the last
layers of networks, which map input representations to class scores.
Given target class $j$, and its classifier $W_j$ and bias $b_j$, let
class $j$'s top $k$ nearest neighbor classifiers be defined as $V_i^j$ ,
$i \in (1, \dots ,K)$. Furthermore, we also define the trivial case
where the immediate nearest neighbor is itself, $V_0^j = W_j$. -->

**Analyzing rare class performance.** We evaluated the predictions of
the \ac{RIDE} model[@ride] on 'few' split test samples in ImageNet-LT.
The 'few' split is comprised of classes with less than 20 training
samples (note that at test time, all classes have 50 samples) and the
\ac{RIDE} model achieves an accuracy of xx.xx% on this split. To
categorize predictions, for each class in the 'few' split of
ImageNet-LT, we found the 10 nearest neighbors in the 'base' split (all
classes not in the 'few' split) using Euclidean distance ($\mu(a, b) =
\left\|{a - b}\right\|$) between features from the \ac{RIDE} model. For
any class, we refer to its nearest neighbors as visually similar
classes.

**AlphaNet implementation.** We will use the term 'classifier' to denote
the linear mapping from feature vectors to class scores. For a Class
$c$, this is a vector $w_c \in R^d$ (in convolutional networks, the last
layer is generally a matrix of all individual classifiers). Given a
feature vector $z = f(I)$ for some image $I$, $w_c^T z$ is the
prediction score for class $c$ (the bias term is omitted here for
simplicity), and the model's class prediction for $I$ is given by
$\argmax_c w_c^T f(I)$.

Given a long-tailed dataset, we will refer to the set of rare classes
(which form the "tail" of the training data distribution) as the 'few'
split, and the set of remaining classes as the 'base' split. AlphaNet is
applied to update the 'few' split classifiers using nearest neighbors
from the 'base' split.

Let $C^F$ and $C^B$ be the set of 'few' and 'base' split classes
respectively. Following the notation described in @sec:method, for a
'few' split class $c \in C^F$, let the $k$ nearest 'base' split
neighbors be $q^c_1, \dots, q^c_k$. Let $w^c$ be the classifier (a
vector in $\R^d$) for $c$, and let $v^c_1, \dots, v^c_k$ be the
classifiers for its nearest neighbors.

## AlphaNet

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
that is, a classifier's change is bound by the norm of its class'
nearest neighbors. Thanks to this, we do not need to update or rescale
'base' split classifiers, which may not be possible in certain domains.

A single network is used to generate coefficients for every 'few' split
class. So, once trained, AlphaNet can be applied even to previously
unseen classes.

## Training {#sec:method:training}

To train AlphaNet, we have to learn parameters $\v{\theta}$ for the
network which maps $\c{\v{v}}^c$ to $\v{\alpha}^c$. We also learn a new
set of bias values for the 'few' split classes, $\tilde{b}_1, \dots,
\tilde{b}_{\abs{C^F}}$. We train AlphaNet using samples from both the
'few' and 'base' splits to prevent over-fitting. So, for a sample $(I,
y)$, the prediction score is given by
$$
s(I, y; \v{\theta}, \tilde{\v{b}}) = \begin{cases}
    f(I)^T \shat{\v{w}}^y + \tilde{b}_y & y \in \mathcal{C}^F \\
    f(I)^T \v{w}^y + b_y                & y \in \mathcal{C}^B
\end{cases}
$$ {#eq:training_probs}
This score is used to compute the softmax cross-entropy loss on a set of
training samples, and minimized with respect to $\v{\theta}$ and
$\tilde{\v{b}}$ using \ac{SGD}.
