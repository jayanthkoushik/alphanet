Parameter                Value
-----------------------  -----------------------------------------------------------
Optimizer                AdamW[@2019.Hutter.Loshchilov] with default parameters
                         ($\beta_1=0.9, \beta_2=0.999, \epsilon=0.01, \lambda=0.01$)
Initial learning rate    $0.001$
Learning rate decay      $0.1$ every 10 epochs
Training epochs          10 (CIFAR‑100‑LT and iNaturalist)
                         25 (ImageNet‑LT and Places‑LT)
Minimum epochs           5
Batch size               256 for iNaturalist, 64 for all others
AlphaNet architecture    3 fully connected layers each with 32 units
Hidden layer activation  Leaky-ReLU[@2011.Bengio.Glorot] with negative slope 0.01
Weight initialization    Uniform sampling with bounds $\pm 1/\sqrt{k}$
                         where $k$ is the number of input units to a layer

: Training hyper-parameters for main experiments. {#tbl:hyperparams}
