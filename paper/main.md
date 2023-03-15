---
title: "AlphaNet: Improving Long-Tail Classification By Combining Classifiers"

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

- name: Yu-Xiong Wang
  affiliation:
  - 2
  email: "yxw@illinois.edu"

- name: Aarti Singh
  affiliation:
  - 1
  email: "aartisingh@cmu.edu"

- name: Martial Hebert
  affiliation:
  - 1
  email: "hebert@cs.cmu.edu"

- name: Michael J. Tarr
  affiliation:
  - 1
  email: "michaeltarr@cmu.edu"

institute:
- id: 1
  name: Carnegie Mellon University

- id: 2
  name: University of Illinois at Urbana-Champaign

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
  representation space and learn weights to update rare classifiers with
  a linear combination of frequent classifiers. AlphaNet, applied on
  several different models, greatly improves test accuracy for rare
  classes in multiple long-tail datasets. We then analyze predictions
  from AlphaNet and find that remaining errors are to often due to
  fine-grained differences among semantically similar classes (e.g., dog
  breeds). Evaluating with semantically similar classes grouped
  together, AlphaNet also improves overall accuracy, showing that the
  method is practical for long-tail classification problems.

# appendices:
# - appendix/main.md
---

{% include utils/debugcmd.md %}

{% include utils/commands.md %}

{% include utils/acronyms.md %}

{% include sections/1_intro.md %}

{% include sections/2_relwork.md %}

{% include sections/3_method.md %}

{% include sections/4_experiments.md %}

{% include sections/5_conclusion.md %}
