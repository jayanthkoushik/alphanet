Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      40.5         54.8         67.4         57.7
AlphaNet (_k_\ =\ 1)   40.4^21.24^  40.2^15.80^  51.9^16.72^  44.7^11.12^
AlphaNet (_k_\ =\ 2)   53.8^ 5.61^  41.6^10.43^  54.6^11.14^  48.3^ 8.52^
AlphaNet (_k_\ =\ 3)   50.9^ 1.87^  49.1^ 0.58^  62.7^ 0.46^  54.6^ 0.32^
AlphaNet (_k_\ =\ 4)   50.6^ 1.61^  49.8^ 0.85^  63.4^ 0.54^  55.1^ 0.42^
AlphaNet (_k_\ =\ 5)   51.3^ 1.09^  50.1^ 0.60^  63.6^ 0.44^  55.5^ 0.39^
AlphaNet (_k_\ =\ 6)   51.1^ 1.29^  50.5^ 0.93^  63.8^ 0.56^  55.7^ 0.49^
AlphaNet (_k_\ =\ 7)   51.5^ 1.43^  50.3^ 0.77^  63.8^ 0.55^  55.6^ 0.44^
AlphaNet (_k_\ =\ 8)   51.6^ 1.00^  50.5^ 0.45^  63.9^ 0.46^  55.8^ 0.29^
AlphaNet (_k_\ =\ 9)   51.1^ 1.61^  50.8^ 0.71^  64.1^ 0.63^  56.0^ 0.37^
AlphaNet (_k_\ =\ 10)  51.1^ 0.70^  51.0^ 0.43^  64.3^ 0.29^  56.2^ 0.25^

: Accuracy computed by considering predictions within 3 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.25_semantic3_adjusted_top1}
