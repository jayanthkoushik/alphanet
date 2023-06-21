Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      37.5         51.9         65.7         55.2
AlphaNet (_k_\ =\ 1)   46.6^20.92^  30.3^15.33^  42.4^16.46^  37.2^10.83^
AlphaNet (_k_\ =\ 2)   44.3^ 7.25^  44.0^10.76^  57.9^11.24^  49.4^ 8.50^
AlphaNet (_k_\ =\ 3)   42.3^ 1.23^  49.3^ 0.56^  63.4^ 0.41^  53.8^ 0.26^
AlphaNet (_k_\ =\ 4)   43.7^ 0.72^  49.5^ 0.42^  63.5^ 0.21^  54.1^ 0.26^
AlphaNet (_k_\ =\ 5)   43.1^ 1.59^  49.5^ 0.46^  63.7^ 0.31^  54.1^ 0.20^
AlphaNet (_k_\ =\ 6)   43.6^ 1.41^  50.0^ 0.33^  63.9^ 0.36^  54.5^ 0.12^
AlphaNet (_k_\ =\ 7)   44.8^ 1.19^  49.6^ 0.37^  63.7^ 0.35^  54.4^ 0.18^
AlphaNet (_k_\ =\ 8)   44.0^ 1.09^  50.0^ 0.34^  63.9^ 0.25^  54.6^ 0.14^
AlphaNet (_k_\ =\ 9)   43.7^ 0.99^  50.1^ 0.29^  64.1^ 0.27^  54.6^ 0.15^
AlphaNet (_k_\ =\ 10)  43.6^ 1.26^  50.1^ 0.44^  64.1^ 0.27^  54.6^ 0.15^

: Accuracy computed by considering predictions within 2 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_1_semantic2_adjusted_top1}
