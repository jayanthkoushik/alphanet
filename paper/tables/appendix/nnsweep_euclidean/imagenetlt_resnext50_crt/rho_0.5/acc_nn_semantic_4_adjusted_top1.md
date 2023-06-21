Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      46.5         59.7         71.0        62.3
AlphaNet (_k_\ =\ 1)   39.2^16.36^  52.4^12.43^  63.1^13.56^  54.7^8.95^
AlphaNet (_k_\ =\ 2)   54.9^ 5.88^  50.2^ 9.54^  61.4^10.37^  55.2^7.77^
AlphaNet (_k_\ =\ 3)   52.4^ 1.35^  55.9^ 0.52^  67.9^ 0.36^  60.1^0.33^
AlphaNet (_k_\ =\ 4)   52.6^ 1.65^  56.4^ 0.77^  68.3^ 0.42^  60.5^0.49^
AlphaNet (_k_\ =\ 5)   52.8^ 0.96^  57.2^ 0.37^  68.8^ 0.28^  61.1^0.18^
AlphaNet (_k_\ =\ 6)   53.1^ 1.13^  56.9^ 0.43^  68.7^ 0.35^  60.9^0.21^
AlphaNet (_k_\ =\ 7)   53.3^ 0.80^  57.1^ 0.43^  68.8^ 0.29^  61.1^0.25^
AlphaNet (_k_\ =\ 8)   52.6^ 0.96^  57.4^ 0.47^  69.1^ 0.30^  61.2^0.25^
AlphaNet (_k_\ =\ 9)   53.4^ 0.64^  57.3^ 0.26^  69.0^ 0.22^  61.3^0.16^
AlphaNet (_k_\ =\ 10)  52.9^ 0.91^  57.5^ 0.37^  69.2^ 0.27^  61.4^0.17^

: Accuracy computed by considering predictions within 4 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.5_semantic4_adjusted_top1}
