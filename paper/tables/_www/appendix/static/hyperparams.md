Parameter                Value
-----------------------  -----------------------------------------------------------
Optimizer                AdamW[@2019.Hutter.Loshchilov] with default parameters
Initial learning rate    0.001
Learning rate decay      0.1 every 10 epochs
Training epochs          10 (CIFAR‑100‑LT and iNaturalist)
                         25 (ImageNet‑LT and Places‑LT)
Minimum epochs           5
Batch size               256 for iNaturalist, 64 for all others
AlphaNet architecture    3 fully connected layers each with 32 units
Hidden layer activation  Leaky-ReLU[@2011.Bengio.Glorot] with negative slope 0.01
Weight initialization    Uniform sampling with bounds +/- (1 / √_m_)
                         where _m_ is the number of input units to a layer

: Training hyper-parameters for main experiments. {#tbl:hyperparams}
