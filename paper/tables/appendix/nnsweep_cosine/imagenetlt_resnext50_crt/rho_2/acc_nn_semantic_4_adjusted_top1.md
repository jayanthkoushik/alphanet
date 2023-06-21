Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      46.5         59.7         71.0        62.3
AlphaNet (_k_\ =\ 1)   42.7^17.69^  49.7^13.42^  60.5^14.34^  52.9^9.54^
AlphaNet (_k_\ =\ 2)   46.8^ 0.76^  58.0^ 0.24^  69.7^ 0.18^  61.0^0.13^
AlphaNet (_k_\ =\ 3)   48.1^ 1.14^  58.1^ 0.44^  69.8^ 0.31^  61.2^0.20^
AlphaNet (_k_\ =\ 4)   48.5^ 1.24^  58.3^ 0.33^  70.0^ 0.28^  61.5^0.17^
AlphaNet (_k_\ =\ 5)   49.1^ 1.23^  58.5^ 0.35^  70.1^ 0.27^  61.7^0.15^
AlphaNet (_k_\ =\ 6)   48.2^ 1.08^  58.7^ 0.26^  70.2^ 0.19^  61.7^0.16^
AlphaNet (_k_\ =\ 7)   49.2^ 0.96^  58.8^ 0.18^  70.2^ 0.23^  61.9^0.08^
AlphaNet (_k_\ =\ 8)   48.7^ 0.92^  58.7^ 0.21^  70.2^ 0.15^  61.8^0.13^
AlphaNet (_k_\ =\ 9)   49.4^ 0.88^  58.7^ 0.20^  70.2^ 0.17^  61.8^0.10^
AlphaNet (_k_\ =\ 10)  48.9^ 1.25^  58.8^ 0.29^  70.3^ 0.23^  61.9^0.09^

: Accuracy computed by considering predictions within 4 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_2_semantic4_adjusted_top1}
