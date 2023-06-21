Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      46.5         59.7         71.0        62.3
AlphaNet (_k_\ =\ 1)   49.4^17.49^  44.7^13.29^  54.7^14.49^  49.2^9.57^
AlphaNet (_k_\ =\ 2)   51.7^ 5.33^  53.4^ 7.92^  64.8^ 8.38^  57.6^6.32^
AlphaNet (_k_\ =\ 3)   50.5^ 0.80^  57.3^ 0.48^  69.0^ 0.28^  60.9^0.24^
AlphaNet (_k_\ =\ 4)   50.7^ 1.14^  57.7^ 0.37^  69.3^ 0.33^  61.2^0.24^
AlphaNet (_k_\ =\ 5)   50.8^ 1.11^  57.9^ 0.45^  69.5^ 0.28^  61.4^0.19^
AlphaNet (_k_\ =\ 6)   51.0^ 0.68^  57.9^ 0.32^  69.5^ 0.17^  61.4^0.17^
AlphaNet (_k_\ =\ 7)   51.2^ 0.88^  58.0^ 0.34^  69.6^ 0.20^  61.5^0.14^
AlphaNet (_k_\ =\ 8)   51.5^ 1.04^  58.0^ 0.29^  69.6^ 0.24^  61.5^0.15^
AlphaNet (_k_\ =\ 9)   51.3^ 1.04^  58.1^ 0.30^  69.7^ 0.26^  61.6^0.13^
AlphaNet (_k_\ =\ 10)  51.0^ 1.27^  58.1^ 0.43^  69.8^ 0.34^  61.6^0.18^

: Accuracy computed by considering predictions within 4 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_1_semantic4_adjusted_top1}
