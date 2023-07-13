Parameter                Value
-----------------------  -------------------------------------------------------
Optimizer                AdamW[@2017.Hutter.Loshchilov] with default parameters:
                             $\beta_1=0.9$
                             $\beta_2=0.999$
                             $\epsilon=0.01$
                             $\lambda=0.01$
Initial learning rate    $0.001$
Learning rate decay      $0.1$ every 10 epochs
Training epochs          10 for CIFAR‑100‑LT, 25 for others
Minimum epochs           5 (not applicable to CIFAR‑100‑LT, see text)
Batch size               64
AlphaNet architecture    3 fully connected layers each with 32 units
Hidden layer activation  Leaky-ReLU[@maas2013rectifier] with negative slope 0.01

: Training hyper-parameters for main experiments. {#tbl:hyperparams}
