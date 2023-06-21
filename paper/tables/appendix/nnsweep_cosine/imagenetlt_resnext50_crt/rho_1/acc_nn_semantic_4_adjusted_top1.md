Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      46.5         59.7         71.0        62.3
AlphaNet (_k_\ =\ 1)   53.0^16.54^  41.9^12.55^  52.1^13.41^  47.4^8.92^
AlphaNet (_k_\ =\ 2)   51.4^ 5.60^  53.2^ 8.67^  64.9^ 8.87^  57.5^6.81^
AlphaNet (_k_\ =\ 3)   49.7^ 0.83^  57.4^ 0.50^  69.2^ 0.33^  60.9^0.25^
AlphaNet (_k_\ =\ 4)   51.0^ 0.71^  57.6^ 0.34^  69.3^ 0.17^  61.2^0.24^
AlphaNet (_k_\ =\ 5)   50.4^ 1.15^  57.6^ 0.40^  69.5^ 0.25^  61.2^0.19^
AlphaNet (_k_\ =\ 6)   50.9^ 1.08^  58.1^ 0.26^  69.6^ 0.30^  61.6^0.09^
AlphaNet (_k_\ =\ 7)   51.9^ 0.92^  57.8^ 0.31^  69.4^ 0.30^  61.5^0.18^
AlphaNet (_k_\ =\ 8)   51.2^ 0.87^  58.1^ 0.28^  69.7^ 0.21^  61.6^0.13^
AlphaNet (_k_\ =\ 9)   50.9^ 0.78^  58.2^ 0.26^  69.8^ 0.21^  61.7^0.13^
AlphaNet (_k_\ =\ 10)  50.8^ 0.87^  58.2^ 0.38^  69.8^ 0.19^  61.7^0.15^

: Accuracy computed by considering predictions within 4 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_1_semantic4_adjusted_top1}
