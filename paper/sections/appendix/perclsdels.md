{% raw %}
\graphicspath{{figures/appendix/}}
{% endraw %}

# Change in per-class accuracies {#sec:perclsdels}

In this section, we analyze the change in accuracy for individual
classes after applying AlphaNet (following the training process
described in @sec:impl). First, we plotted the sorted accuracy changes,
grouped by split. These are shown in the following figures:

* ImageNet‑LT:
    * cRT baseline: @fig:rhos_cls_deltas_imagenetlt_crt.
    * LWS baseline: @fig:rhos_cls_deltas_imagenetlt_lws.
    * RIDE baseline: @fig:rhos_cls_deltas_imagenetlt_ride.

* Places‑LT:
    * cRT baseline: @fig:rhos_cls_deltas_placeslt_crt.
    * LWS baseline: @fig:rhos_cls_deltas_placeslt_lws.

* CIFAR‑100‑LT:
    * RIDE baseline: @fig:rhos_cls_deltas_cifarlt_ride.
    * LTR baseline: @fig:rhos_cls_deltas_cifarlt_ltr.

We also plotted accuracy change for classes against the average distance
to their 5 nearest neighbors by Euclidean distance. For 'few' split
classes, we selected neighbors from the 'base' split, and for the 'base'
split classes, we selected neighbors from the 'few' split. Recall that
classifiers for the 'few' split were updated using classifiers from
these neighbors. Results are shown in the following figures:

* ImageNet‑LT:
    * cRT baseline: @fig:rhos_cls_delta_vs_nndist_imagenetlt:crt.
    * LWS baseline: @fig:rhos_cls_delta_vs_nndist_imagenetlt:lws.
    * RIDE baseline: @fig:rhos_cls_delta_vs_nndist_imagenetlt:ride.

* Places‑LT:
    * cRT baseline: @fig:rhos_cls_delta_vs_nndist_placeslt:crt.
    * LWS baseline: @fig:rhos_cls_delta_vs_nndist_placeslt:lws.

* CIFAR‑100‑LT:
    * RIDE baseline: @fig:rhos_cls_delta_vs_nndist_cifarlt:ride.
    * LTR baseline: @fig:rhos_cls_delta_vs_nndist_cifarlt:ltr.

\clearpage

![Change in per-class test accuracy on ImageNet‑LT after AlphaNet
training with cRT baseline. Each bar shows the change the change in
accuracy for one class. The solid lines in each split show the average
per-class change for the split, and the dotted line shows the overall
average per-class
change.](figures/appendix/rhos_cls_deltas_imagenetlt_crt){#fig:rhos_cls_deltas_imagenetlt_crt}

\clearpage

![Change in per-class test accuracy on ImageNet‑LT after AlphaNet
training with LWS baseline. Each bar shows the change the change in
accuracy for one class. The solid lines in each split show the average
per-class change for the split, and the dotted line shows the overall
average per-class
change.](figures/appendix/rhos_cls_deltas_imagenetlt_lws){#fig:rhos_cls_deltas_imagenetlt_lws}

\clearpage

![Change in per-class test accuracy on ImageNet‑LT after AlphaNet
training with RIDE baseline. Each bar shows the change the change in
accuracy for one class. The solid lines in each split show the average
per-class change for the split, and the dotted line shows the overall
average per-class
change.](figures/appendix/rhos_cls_deltas_imagenetlt_ride){#fig:rhos_cls_deltas_imagenetlt_ride}

\clearpage

![Change in per-class test accuracy on Places‑LT after AlphaNet training
with cRT baseline. Each bar shows the change the change in accuracy for
one class. The solid lines in each split show the average per-class
change for the split, and the dotted line shows the overall average
per-class
change.](figures/appendix/rhos_cls_deltas_placeslt_crt){#fig:rhos_cls_deltas_placeslt_crt}

\clearpage

![Change in per-class test accuracy on Places‑LT after AlphaNet training
with LWS baseline. Each bar shows the change the change in accuracy for
one class. The solid lines in each split show the average per-class
change for the split, and the dotted line shows the overall average
per-class
change.](figures/appendix/rhos_cls_deltas_placeslt_lws){#fig:rhos_cls_deltas_placeslt_lws}

\clearpage

![Change in per-class test accuracy on CIFAR‑100‑LT after AlphaNet
training with RIDE baseline. Each bar shows the change the change in
accuracy for one class. The solid lines in each split show the average
per-class change for the split, and the dotted line shows the overall
average per-class
change.](figures/appendix/rhos_cls_deltas_cifarlt_ride){#fig:rhos_cls_deltas_cifarlt_ride}

\clearpage

![Change in per-class test accuracy on CIFAR‑100‑LT after AlphaNet
training with LTR baseline. Each bar shows the change the change in
accuracy for one class. The solid lines in each split show the average
per-class change for the split, and the dotted line shows the overall
average per-class
change.](figures/appendix/rhos_cls_deltas_cifarlt_ltr){#fig:rhos_cls_deltas_cifarlt_ltr}

\clearpage

<div id="fig:rhos_cls_delta_vs_nndist_imagenetlt">

![cRT
baseline](figures/appendix/rhos_cls_delta_vs_nndist_imagenetlt_crt){#fig:rhos_cls_delta_vs_nndist_imagenetlt:crt}

![LWS
baseline](figures/appendix/rhos_cls_delta_vs_nndist_imagenetlt_lws){#fig:rhos_cls_delta_vs_nndist_imagenetlt:lws}

![RIDE
baseline](figures/appendix/rhos_cls_delta_vs_nndist_imagenetlt_ride){#fig:rhos_cls_delta_vs_nndist_imagenetlt:ride}

Change in per-class test accuracy on ImageNet‑LT, versus mean distance
to 5 nearest neighbors based on Euclidean distance. The neighbors are
from 'base' split for the 'few' split classes, and vice-versa for the
'base' split classes. The lines are regression fits, and the $r$ values
are Pearson correlations.

</div>

\clearpage

<div id="fig:rhos_cls_delta_vs_nndist_placeslt">

![cRT
baseline](figures/appendix/rhos_cls_delta_vs_nndist_placeslt_crt){#fig:rhos_cls_delta_vs_nndist_placeslt:crt}

![LWS
baseline](figures/appendix/rhos_cls_delta_vs_nndist_placeslt_lws){#fig:rhos_cls_delta_vs_nndist_placeslt:lws}

Change in per-class test accuracy on Places‑LT, versus mean distance to
5 nearest neighbors based on Euclidean distance. The neighbors are from
'base' split for the 'few' split classes, and vice-versa for the 'base'
split classes. The lines are regression fits, and the $r$ values are
Pearson correlations.

</div>

\clearpage

<div id="fig:rhos_cls_delta_vs_nndist_cifarlt">

![RIDE
baseline](figures/appendix/rhos_cls_delta_vs_nndist_cifarlt_ride){#fig:rhos_cls_delta_vs_nndist_cifarlt:ride}

![LTR
baseline](figures/appendix/rhos_cls_delta_vs_nndist_cifarlt_ltr){#fig:rhos_cls_delta_vs_nndist_cifarlt:ltr}

Change in per-class test accuracy on CIFAR‑100‑LT, versus mean distance
to 5 nearest neighbors based on Euclidean distance. The neighbors are
from 'base' split for the 'few' split classes, and vice-versa for the
'base' split classes. The lines are regression fits, and the $r$ values
are Pearson correlations.

</div>

\clearpage
