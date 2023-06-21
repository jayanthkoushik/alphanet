Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      37.5         51.9         65.7         55.2
AlphaNet (_k_\ =\ 1)   42.2^22.21^  33.6^16.27^  45.5^17.84^  39.4^11.64^
AlphaNet (_k_\ =\ 2)   44.7^ 6.92^  44.3^ 9.77^  57.9^10.46^  49.6^ 7.80^
AlphaNet (_k_\ =\ 3)   43.2^ 1.14^  49.2^ 0.55^  63.2^ 0.36^  53.8^ 0.25^
AlphaNet (_k_\ =\ 4)   43.2^ 1.58^  49.6^ 0.50^  63.5^ 0.41^  54.1^ 0.29^
AlphaNet (_k_\ =\ 5)   43.5^ 1.43^  49.8^ 0.52^  63.8^ 0.34^  54.3^ 0.22^
AlphaNet (_k_\ =\ 6)   43.8^ 0.94^  49.8^ 0.37^  63.8^ 0.24^  54.3^ 0.21^
AlphaNet (_k_\ =\ 7)   44.0^ 1.04^  49.9^ 0.40^  63.9^ 0.26^  54.5^ 0.17^
AlphaNet (_k_\ =\ 8)   44.4^ 1.31^  49.8^ 0.33^  63.8^ 0.30^  54.5^ 0.17^
AlphaNet (_k_\ =\ 9)   44.1^ 1.35^  49.9^ 0.35^  64.0^ 0.32^  54.5^ 0.13^
AlphaNet (_k_\ =\ 10)  43.8^ 1.63^  50.0^ 0.50^  64.1^ 0.46^  54.6^ 0.21^

: Accuracy computed by considering predictions within 2 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_1_semantic2_adjusted_top1}
