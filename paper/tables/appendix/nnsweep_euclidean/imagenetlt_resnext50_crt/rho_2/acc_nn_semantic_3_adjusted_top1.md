Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      40.5         54.8         67.4         57.7
AlphaNet (_k_\ =\ 1)   36.4^20.85^  43.3^15.57^  55.2^16.50^  46.9^10.98^
AlphaNet (_k_\ =\ 2)   41.5^ 1.33^  52.8^ 0.56^  65.7^ 0.30^  56.2^ 0.21^
AlphaNet (_k_\ =\ 3)   42.3^ 1.21^  53.2^ 0.33^  66.0^ 0.33^  56.7^ 0.14^
AlphaNet (_k_\ =\ 4)   42.8^ 1.47^  53.4^ 0.39^  66.3^ 0.29^  56.9^ 0.19^
AlphaNet (_k_\ =\ 5)   44.4^ 1.07^  53.2^ 0.33^  66.2^ 0.22^  57.0^ 0.14^
AlphaNet (_k_\ =\ 6)   43.4^ 1.09^  53.7^ 0.30^  66.5^ 0.16^  57.2^ 0.09^
AlphaNet (_k_\ =\ 7)   43.7^ 1.05^  53.7^ 0.27^  66.5^ 0.20^  57.3^ 0.14^
AlphaNet (_k_\ =\ 8)   43.7^ 1.26^  53.7^ 0.34^  66.6^ 0.23^  57.3^ 0.11^
AlphaNet (_k_\ =\ 9)   43.9^ 0.89^  53.8^ 0.26^  66.6^ 0.15^  57.4^ 0.08^
AlphaNet (_k_\ =\ 10)  43.9^ 1.31^  53.8^ 0.35^  66.6^ 0.31^  57.4^ 0.12^

: Accuracy computed by considering predictions within 3 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_2_semantic3_adjusted_top1}
