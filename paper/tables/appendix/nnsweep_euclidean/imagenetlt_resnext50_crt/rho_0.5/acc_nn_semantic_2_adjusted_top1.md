Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      37.5         51.9         65.7         55.2
AlphaNet (_k_\ =\ 1)   29.3^20.77^  43.1^15.22^  55.9^16.69^  46.1^10.89^
AlphaNet (_k_\ =\ 2)   48.8^ 7.53^  40.3^11.93^  53.6^12.99^  46.6^ 9.69^
AlphaNet (_k_\ =\ 3)   45.8^ 1.72^  47.4^ 0.54^  61.8^ 0.43^  52.7^ 0.37^
AlphaNet (_k_\ =\ 4)   46.0^ 1.94^  48.0^ 0.93^  62.3^ 0.53^  53.2^ 0.56^
AlphaNet (_k_\ =\ 5)   46.2^ 1.21^  48.8^ 0.41^  63.0^ 0.29^  53.9^ 0.16^
AlphaNet (_k_\ =\ 6)   46.5^ 1.46^  48.5^ 0.47^  62.8^ 0.38^  53.7^ 0.21^
AlphaNet (_k_\ =\ 7)   46.6^ 0.97^  48.7^ 0.50^  62.9^ 0.36^  53.9^ 0.29^
AlphaNet (_k_\ =\ 8)   45.8^ 1.18^  49.2^ 0.50^  63.3^ 0.34^  54.1^ 0.24^
AlphaNet (_k_\ =\ 9)   46.9^ 0.67^  49.0^ 0.31^  63.2^ 0.28^  54.2^ 0.19^
AlphaNet (_k_\ =\ 10)  46.2^ 1.22^  49.2^ 0.41^  63.4^ 0.36^  54.3^ 0.19^

: Accuracy computed by considering predictions within 2 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.5_semantic2_adjusted_top1}
