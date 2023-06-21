Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      46.5         59.7         71.0        62.3
AlphaNet (_k_\ =\ 1)   46.2^18.05^  47.1^13.69^  57.7^14.63^  51.1^9.74^
AlphaNet (_k_\ =\ 2)   52.8^ 4.21^  54.1^ 6.44^  65.7^ 6.95^  58.4^5.20^
AlphaNet (_k_\ =\ 3)   52.3^ 1.02^  56.2^ 0.72^  68.0^ 0.47^  60.2^0.41^
AlphaNet (_k_\ =\ 4)   52.5^ 1.15^  56.4^ 0.36^  68.4^ 0.27^  60.5^0.27^
AlphaNet (_k_\ =\ 5)   52.8^ 1.44^  56.7^ 0.41^  68.6^ 0.36^  60.7^0.22^
AlphaNet (_k_\ =\ 6)   53.2^ 0.84^  57.2^ 0.48^  68.8^ 0.34^  61.1^0.26^
AlphaNet (_k_\ =\ 7)   53.6^ 0.77^  57.1^ 0.39^  68.8^ 0.26^  61.1^0.20^
AlphaNet (_k_\ =\ 8)   54.0^ 0.82^  57.0^ 0.42^  68.7^ 0.33^  61.1^0.26^
AlphaNet (_k_\ =\ 9)   53.4^ 0.62^  57.3^ 0.24^  69.0^ 0.22^  61.3^0.16^
AlphaNet (_k_\ =\ 10)  53.5^ 0.62^  57.3^ 0.31^  69.0^ 0.22^  61.3^0.19^

: Accuracy computed by considering predictions within 4 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.5_semantic4_adjusted_top1}
