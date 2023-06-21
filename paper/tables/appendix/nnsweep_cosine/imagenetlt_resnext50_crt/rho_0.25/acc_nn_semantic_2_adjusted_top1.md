Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      37.5         51.9         65.7         55.2
AlphaNet (_k_\ =\ 1)   29.3^20.92^  43.0^15.33^  56.0^16.46^  46.2^10.83^
AlphaNet (_k_\ =\ 2)   49.7^ 5.37^  41.5^ 9.71^  55.3^10.54^  47.9^ 7.98^
AlphaNet (_k_\ =\ 3)   49.9^ 4.19^  43.1^ 7.67^  57.2^ 8.24^  49.4^ 6.31^
AlphaNet (_k_\ =\ 4)   48.7^ 1.65^  46.4^ 0.68^  61.1^ 0.61^  52.3^ 0.51^
AlphaNet (_k_\ =\ 5)   48.8^ 2.11^  47.3^ 0.84^  61.5^ 0.78^  53.0^ 0.45^
AlphaNet (_k_\ =\ 6)   50.2^ 1.50^  46.8^ 0.86^  61.2^ 0.67^  52.8^ 0.47^
AlphaNet (_k_\ =\ 7)   49.3^ 1.23^  47.3^ 0.86^  61.7^ 0.66^  53.1^ 0.52^
AlphaNet (_k_\ =\ 8)   49.6^ 1.73^  47.2^ 0.80^  61.7^ 0.72^  53.1^ 0.44^
AlphaNet (_k_\ =\ 9)   49.3^ 0.91^  47.6^ 0.61^  61.9^ 0.55^  53.3^ 0.41^
AlphaNet (_k_\ =\ 10)  49.5^ 0.98^  47.3^ 0.45^  61.7^ 0.36^  53.2^ 0.30^

: Accuracy computed by considering predictions within 2 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.25_semantic2_adjusted_top1}
