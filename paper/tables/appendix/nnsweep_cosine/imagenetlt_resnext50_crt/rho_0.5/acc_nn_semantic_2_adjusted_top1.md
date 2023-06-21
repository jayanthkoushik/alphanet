Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      37.5         51.9         65.7         55.2
AlphaNet (_k_\ =\ 1)   38.0^22.83^  36.7^16.72^  49.2^17.96^  41.7^11.82^
AlphaNet (_k_\ =\ 2)   46.1^ 5.34^  45.1^ 8.04^  58.9^ 8.67^  50.5^ 6.47^
AlphaNet (_k_\ =\ 3)   45.7^ 1.44^  47.7^ 0.91^  61.9^ 0.62^  52.9^ 0.50^
AlphaNet (_k_\ =\ 4)   45.9^ 1.24^  47.9^ 0.42^  62.3^ 0.34^  53.2^ 0.29^
AlphaNet (_k_\ =\ 5)   46.3^ 1.85^  48.4^ 0.58^  62.6^ 0.51^  53.6^ 0.28^
AlphaNet (_k_\ =\ 6)   46.4^ 1.13^  49.0^ 0.55^  63.0^ 0.37^  54.0^ 0.27^
AlphaNet (_k_\ =\ 7)   47.1^ 0.87^  48.8^ 0.42^  62.9^ 0.37^  54.0^ 0.26^
AlphaNet (_k_\ =\ 8)   47.5^ 0.95^  48.6^ 0.52^  62.8^ 0.38^  53.9^ 0.31^
AlphaNet (_k_\ =\ 9)   46.8^ 0.80^  48.9^ 0.35^  63.1^ 0.30^  54.1^ 0.21^
AlphaNet (_k_\ =\ 10)  46.9^ 0.74^  49.1^ 0.41^  63.2^ 0.29^  54.2^ 0.24^

: Accuracy computed by considering predictions within 2 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.5_semantic2_adjusted_top1}
