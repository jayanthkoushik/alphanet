Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      40.5         54.8         67.4         57.7
AlphaNet (_k_\ =\ 1)   40.4^21.38^  40.2^15.96^  52.2^16.68^  44.9^11.16^
AlphaNet (_k_\ =\ 2)   48.2^ 5.01^  48.3^ 7.69^  61.3^ 8.10^  53.3^ 6.13^
AlphaNet (_k_\ =\ 3)   47.8^ 1.30^  50.8^ 0.91^  64.1^ 0.50^  55.5^ 0.47^
AlphaNet (_k_\ =\ 4)   47.9^ 1.20^  51.0^ 0.38^  64.5^ 0.31^  55.8^ 0.25^
AlphaNet (_k_\ =\ 5)   48.4^ 1.73^  51.4^ 0.51^  64.7^ 0.42^  56.1^ 0.22^
AlphaNet (_k_\ =\ 6)   48.7^ 1.10^  52.0^ 0.53^  65.1^ 0.38^  56.6^ 0.26^
AlphaNet (_k_\ =\ 7)   49.2^ 0.81^  51.8^ 0.43^  65.0^ 0.32^  56.5^ 0.24^
AlphaNet (_k_\ =\ 8)   49.6^ 0.92^  51.6^ 0.52^  64.9^ 0.39^  56.5^ 0.31^
AlphaNet (_k_\ =\ 9)   48.9^ 0.73^  52.0^ 0.30^  65.2^ 0.24^  56.7^ 0.17^
AlphaNet (_k_\ =\ 10)  49.0^ 0.69^  52.1^ 0.37^  65.2^ 0.26^  56.8^ 0.21^

: Accuracy computed by considering predictions within 3 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.5_semantic3_adjusted_top1}
