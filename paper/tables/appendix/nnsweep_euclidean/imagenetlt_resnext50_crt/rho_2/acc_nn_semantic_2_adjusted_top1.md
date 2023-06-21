Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      37.5         51.9         65.7         55.2
AlphaNet (_k_\ =\ 1)   33.6^22.21^  39.9^16.27^  52.4^17.84^  43.9^11.64^
AlphaNet (_k_\ =\ 2)   38.9^ 1.49^  49.9^ 0.54^  63.7^ 0.34^  53.7^ 0.18^
AlphaNet (_k_\ =\ 3)   39.7^ 1.25^  50.3^ 0.29^  64.1^ 0.33^  54.1^ 0.14^
AlphaNet (_k_\ =\ 4)   40.3^ 1.55^  50.4^ 0.40^  64.4^ 0.30^  54.4^ 0.20^
AlphaNet (_k_\ =\ 5)   41.9^ 1.21^  50.2^ 0.34^  64.3^ 0.28^  54.5^ 0.16^
AlphaNet (_k_\ =\ 6)   40.8^ 1.22^  50.7^ 0.29^  64.5^ 0.19^  54.7^ 0.08^
AlphaNet (_k_\ =\ 7)   41.2^ 1.09^  50.7^ 0.28^  64.6^ 0.22^  54.8^ 0.14^
AlphaNet (_k_\ =\ 8)   41.2^ 1.38^  50.8^ 0.35^  64.7^ 0.25^  54.8^ 0.11^
AlphaNet (_k_\ =\ 9)   41.3^ 0.94^  50.8^ 0.25^  64.6^ 0.17^  54.8^ 0.07^
AlphaNet (_k_\ =\ 10)  41.4^ 1.38^  50.8^ 0.36^  64.7^ 0.33^  54.9^ 0.12^

: Accuracy computed by considering predictions within 2 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_2_semantic2_adjusted_top1}
