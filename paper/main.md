---
title: "A Simple Strategy to Address Imbalanced Long-tail Classification Accuracies"

author:
  - name: Nadine Chang
    affiliation:
    - 1
    email: "nchang1@cs.cmu.edu"
    equalcontrib: true
  - name: Jayanth Koushik
    affiliation:
    - 1
    email: "jkoushik@andrew.cmu.edu"
    equalcontrib: true
    corresponding: true
  - name: Michael J. Tarr
    affiliation:
    - 1
    email: "michaeltarr@cmu.edu"
  - name: Martial Hebert
    affiliation:
    - 1
    email: "hebert@cs.cmu.edu"
  - name: Yu-Xiong Wang
    affiliation:
    - 2
    email: "yxw@illinois.edu"

institute:
  - id: 1
    name: Carnegie Mellon University
  - id: 2
    name: University of Illinois at Urbana-Champaign

abstract: Methods in long-tail learning focus on improving performance for
  data-poor (rare) classes; however, performance for such classes remains much
  lower than performance for more data-rich (frequent) classes. This is a
  problem for applications such as autonomous vehicle systems or medical
  imaging, where balanced performance across all classes is desirable. We
  propose a simple method that can be applied to existing models, performing a
  _post hoc_ correction, which leads to much higher accuracy on rare classes
  with negligible change to overall accuracy. Starting with a pre-trained model,
  we find frequent classes that are closest to rare classes in the model's
  representation space and use their classifiers in linear combination,
  improving rare class accuracy while preserving overall accuracy. This
  straightforward method is extremely fast to train and can be applied to any
  model without requiring changes to the base classifiers which may be
  restricted for a variety of reasons (e.g., ethical considerations, privacy
  concerns, or intellectual property rights). We observe that our simple
  correction improves performance for rare classes by up to 10 points on the
  ImageNet-LT dataset. At the same time, overall accuracy is preserved, leading
  to more balanced performance across all classes.

appendices:
- 'sections/a_appendix.md'
---

\acrodef{LWS}{learnable weight scaling}
\acrodef{cRT}{classifier re-training}

{% include sections/1_intro.md %}

{% include sections/2_relwork.md %}

{% include sections/3_method.md %}

{% include sections/4_experiments.md %}

{% include sections/5_conclusion.md %}
