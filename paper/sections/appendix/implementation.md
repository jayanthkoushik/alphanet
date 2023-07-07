# Implementation details {#sec:impl_details}

<!-- cSpell:ignore jayanthkoushik, matplotlib, seaborn -->

Experiments were run using the PyTorch[@paszke2019pytorch] library. We
used the container implementation provided by
\ac{NGC}[^note:container_link]. Code to reproduce experimental results
is available on GitHub at
[`github.com/jayanthkoushik/alphanet`](https://github.com/jayanthkoushik/alphanet),
along with instructions to run the code.

Hyper-parameter settings used during training are shown in
@tbl:hyperparams. The selection of the best model was controlled by the
'minimum epochs' parameter. After training for at least this many
epochs, model weights were saved at the end of each epoch. Finally, the
best model was selected based on overall validation accuracy, and used
for testing.

All experiments were repeated 10 times from different random
initializations, and unless specified otherwise, results (e.g.,
accuracy) are average values. In tables, the standard deviation is shown
in superscript.

Plots were generated with Matplotlib[@hunter2007matplotlib] using the
Seaborn library[@waskom2021]. Wherever applicable, error bars show 95%
confidence intervals, estimated using 10,000 bootstrap resamples.

Parameter                       Value
-------------                   --------
Optimizer                       AdamW[@2017.Hutter.Loshchilov] with default parameters:
                                    $\beta_1=0.9$
                                    $\beta_2=0.999$
                                    $\epsilon=0.01$
                                    $\lambda=0.01$
Initial learning rate           $0.001$
Learning rate decay             $0.1$ every 10 epochs
Training epochs                 25
Minimum epochs                  5
Batch size                      64
AlphaNet architecture           3 fully connected layers each with 32 units
Hidden layer activation         Leaky ReLU with negative slope $0.01$

: Hyper-parameters. {#tbl:hyperparams}

[^note:container_link]: Version 22.06 from [catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch).
