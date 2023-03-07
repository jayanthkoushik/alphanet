# Method {#sec:method}

Given a long-tailed dataset, we will refer to the set of rare classes
(which form the "tail" of the training data distribution) as the
'few' split, and the set of remaining classes as the 'base' split. AlphaNet
is applied to update the 'few' split classifiers using nearest neighbors from
the 'base' split.

Let $C^F$ and $C^B$ be the set of 'few' and 'base' split classes respectively.
Following the notation described in @sec:intro:analysis, for a 'few'
split class $c \in C^F$, let the $k$ nearest 'base' split neighbors be
$q^c_1, \dots, q^c_k$. Let $w^c$ be the classifier (a vector in $\R^d$) for $c$,
and let $v^c_1, \dots, v^c_k$ be the classifiers for its nearest neighbors.

<!-- OLD -->

<!-- Long-tail classification is uniquely characterized by its unbalanced,
long-tailed distribution of data. Due to the lack of sufficient data in
parts of the distribution, otherwise known as the "tail", training on
the tail classes significantly suffers compared to that of 'base'
classes that have sufficient data. As a result, tail classes typically
see low accuracy performance, and this remains the dominating challenge
of long-tailed recognition. AlphaNet directly addresses this issue, and
successfully improves the tail performance by combining classifiers in
order to adaptively adjust weak classifiers. -->

## Notation

We first group up our classes into two broad splits: our 'base' split
with $B$ number of classes and our 'few' split with $F$ classes. The
'base' classes contain classes with many examples, and conversely the
'few' classes contain classes with few examples. We denote by $N = B +
F$ the total number of classes.

Additionally, we denote a generic sample by $I$ and its corresponding
label by $y$. We split the training set into two subsets of pairs
$(I_i,y_i)$: the subset $X_{base}$ contains samples from the $B$
classes, and the subset $X_{few}$ contains samples from the $F$ classes.

Finally, we work with a fixed pre-trained model with classifiers $W_j$
and biases $b_j$, $j \in (1, \dots, N)$; here, classifiers are the last
layers of networks, which map input representations to class scores.
Given target class $j$, and its classifier $W_j$ and bias $b_j$, let
class $j$'s top $k$ nearest neighbor classifiers be defined as $V_i^j$ ,
$i \in (1, \dots ,K)$. Furthermore, we also define the trivial case
where the immediate nearest neighbor is itself, $V_0^j = W_j$.

## AlphaNet

Our combination model, AlphaNet, takes in trained classifiers from the
$B$ 'base' classes and weak classifiers from the $F$ 'few' classes, in
order to learn a new classifier composition for each 'few' class. The
following section will detail the model and our training as depicted in
@fig:alphanet.

![An illustration of AlphaNet. The classifier for a 'few' split class,
along with the classifiers of $k$ nearest neighbors are passed through a
network, which outputs $k + 1$ weights, termed alphas. These alphas are
used to linearly combine the input classifiers, producing the updated
classifier for the given 'few' split
class.](figures/model.png){#fig:alphanet}

## Architecture

AlphaNet is lightweight and uses a small fully connected network. Its
input is a flattened concatenated vector of $V_k^i$, $k \in(0, \dots
,K)$, where $V_0^i = W_i$, the original classifier for class $i$. Its
output is an $\alpha$ vector, where $\alpha_k$, $k \in(0, \dots ,K)$
corresponds to input $V_k^i$. Importantly, AlphaNet does not restrict
the range that $\alpha$ coefficients can be. Thus, AlphaNet learns both
positive and negative $\alpha$ coefficients. Furthermore, AlphaNet can
be used to learn all $\alpha$ coefficients for all classifiers. Once
trained, AlphaNet can also be applied on new classes not seen during
training.

## Training

Given alphas $\alpha_k$, and input $V_k^i$, $k \in (0, \dots, K)$, the
updated classifier is given by

$$
U_i = V_0 + \sum_{j=1}^{K} \alpha_j \cdot V_j^i.
$$ {#eq:sumV}

We use the same set of alphas to learn $U_i$, $i \in (1, \dots, F)$ for
each target class.

$$
\begin{aligned}
&U_1 = V_0^1 + \sum_{j=1}^K \alpha_j \cdot V_j^1.\\
&U_F = V_0^F \sum_{j=1}^K \alpha_j \cdot V_j^F.
\end{aligned}
$$ {#eq:allvs}

It is important to note that while we are attempting to compose new
'few' classifiers, we are still working on a long-tailed recognition
problem. Thus, when training we must also consider our 'base'
classifiers that have been sufficient trained. Now, given a training
image sample and label pair $\{I_i, y_i\}$ and its final feature
representation $x_i$, we compute our sample score as

$$
\begin{aligned}
s_{if} &= x_i \cdot U_f. \qquad f \in (1,\dots,F)\\
s_{ib} &= x_i \cdot W_b + b_b. \qquad b \in (1,\dots,B)
\end{aligned}
$$ {#eq:ses}

We combine the two sets of scores and obtain per-class prediction
scores. Finally, we compute the softmax cross entropy loss, which is
minimized to learn AlphaNet weights.
