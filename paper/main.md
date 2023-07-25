---
title: "AlphaNet: Improving Long-Tail Classification By Combining Classifiers"

author:
- name: Nadine Chang
  affiliation:
  - $\bm{\dag}$
  email: "nchang1@cs.cmu.edu"
  equalcontrib: true

- name: Jayanth Koushik
  affiliation:
  - $\bm{\dag}$
  email: "jkoushik@andrew.cmu.edu"
  equalcontrib: true
  corresponding: true

- name: Aarti Singh
  affiliation:
  - $\bm{\dag}$
  email: "aartisingh@cmu.edu"

- name: Martial Hebert
  affiliation:
  - $\bm{\dag}$
  email: "hebert@cs.cmu.edu"

- name: Yu-Xiong Wang
  affiliation:
  - $\bm{\ddag}$
  email: "yxw@illinois.edu"

- name: Michael J. Tarr
  affiliation:
  - $\bm{\dag}$
  email: "michaeltarr@cmu.edu"

institute:
- id: $\bm{\dag}$
  name: Carnegie Mellon University

- id: $\bm{\ddag}$
  name: University of Illinois Urbana-Champaign

abstract: Methods in long-tail learning focus on improving performance
  for data-poor (rare) classes; however, performance for such classes
  remains much lower than performance for more data-rich (frequent)
  classes. Analyzing the predictions of long-tail methods for rare
  classes reveals that a large number of errors are due to
  misclassification of rare items as visually similar frequent classes.
  To address this problem, we introduce AlphaNet, a method that can be
  applied to existing models, performing post hoc correction on
  classifiers of rare classes. Starting with a pre-trained model, we
  find frequent classes that are closest to rare classes in the model's
  representation space and learn weights to update rare class
  classifiers with a linear combination of frequent class classifiers.
  AlphaNet, applied on several models, greatly improves test accuracy
  for rare classes in multiple long-tailed datasets, with very little
  change to the overall accuracy. Our method also provides a way to
  control the trade-off between rare class and overall accuracy, making
  it practical for long-tail classification in the wild.

bibliography: references.bib

includes:
- utils/commands.md
- utils/debugcmd.md

sections:
- sections/1_intro.md
- sections/2_related.md
- sections/3_method.md
- sections/4_experiments.md
- sections/5_conclusion.md
- sections/6_acknowledgements.md

# cSpell: ignore rhosweep, ksweep, perclsdels

appendices:
- sections/appendix/implementation.md
- sections/appendix/ksweep.md
- sections/appendix/rhosweep.md
- sections/appendix/perclsdels.md

url: https://jkoushik.me/alphanet
---

{% raw %}
\graphicspath{{figures/}}
{% endraw %}

\raggedbottom
