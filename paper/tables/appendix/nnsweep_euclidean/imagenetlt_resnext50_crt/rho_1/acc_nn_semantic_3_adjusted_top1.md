Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      40.5         54.8         67.4         57.7
AlphaNet (_k_\ =\ 1)   44.4^20.85^  37.3^15.57^  48.8^16.50^  42.7^10.98^
AlphaNet (_k_\ =\ 2)   47.0^ 6.47^  47.5^ 9.38^  60.3^ 9.75^  52.4^ 7.40^
AlphaNet (_k_\ =\ 3)   45.5^ 1.06^  52.2^ 0.57^  65.2^ 0.30^  56.3^ 0.25^
AlphaNet (_k_\ =\ 4)   45.6^ 1.50^  52.5^ 0.48^  65.6^ 0.38^  56.6^ 0.26^
AlphaNet (_k_\ =\ 5)   45.9^ 1.33^  52.8^ 0.51^  65.7^ 0.33^  56.8^ 0.21^
AlphaNet (_k_\ =\ 6)   46.1^ 0.88^  52.8^ 0.36^  65.8^ 0.19^  56.9^ 0.18^
AlphaNet (_k_\ =\ 7)   46.3^ 0.96^  52.9^ 0.37^  65.9^ 0.23^  57.0^ 0.15^
AlphaNet (_k_\ =\ 8)   46.7^ 1.21^  52.8^ 0.33^  65.8^ 0.27^  57.0^ 0.15^
AlphaNet (_k_\ =\ 9)   46.4^ 1.25^  53.0^ 0.36^  66.0^ 0.30^  57.1^ 0.13^
AlphaNet (_k_\ =\ 10)  46.1^ 1.51^  53.0^ 0.48^  66.1^ 0.39^  57.1^ 0.19^

: Accuracy computed by considering predictions within 3 WordNet nodes as
correct, for AlphaNet using varying number of nearest neighbors (_k_) based on
Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_1_semantic3_adjusted_top1}
