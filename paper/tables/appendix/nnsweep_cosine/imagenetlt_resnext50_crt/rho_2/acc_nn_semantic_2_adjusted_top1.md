Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      37.5         51.9         65.7         55.2
AlphaNet (_k_\ =\ 1)   33.6^22.36^  39.8^16.38^  52.6^17.60^  43.9^11.58^
AlphaNet (_k_\ =\ 2)   38.3^ 0.89^  50.0^ 0.32^  64.0^ 0.24^  53.8^ 0.15^
AlphaNet (_k_\ =\ 3)   40.0^ 1.47^  50.1^ 0.54^  64.1^ 0.38^  54.1^ 0.21^
AlphaNet (_k_\ =\ 4)   40.6^ 1.55^  50.3^ 0.41^  64.4^ 0.32^  54.4^ 0.18^
AlphaNet (_k_\ =\ 5)   41.2^ 1.73^  50.5^ 0.47^  64.4^ 0.33^  54.6^ 0.16^
AlphaNet (_k_\ =\ 6)   40.1^ 1.36^  50.8^ 0.33^  64.7^ 0.26^  54.7^ 0.18^
AlphaNet (_k_\ =\ 7)   41.3^ 1.32^  50.8^ 0.24^  64.6^ 0.29^  54.8^ 0.10^
AlphaNet (_k_\ =\ 8)   40.7^ 1.11^  50.8^ 0.24^  64.7^ 0.18^  54.8^ 0.12^
AlphaNet (_k_\ =\ 9)   41.7^ 1.11^  50.8^ 0.22^  64.6^ 0.22^  54.9^ 0.10^
AlphaNet (_k_\ =\ 10)  41.0^ 1.61^  50.9^ 0.34^  64.8^ 0.31^  54.9^ 0.09^

: Accuracy computed by considering predictions within 2 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_2_semantic2_adjusted_top1}
