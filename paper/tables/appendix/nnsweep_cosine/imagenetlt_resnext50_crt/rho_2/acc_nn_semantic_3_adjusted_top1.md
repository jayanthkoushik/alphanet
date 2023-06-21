Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      40.5         54.8         67.4         57.7
AlphaNet (_k_\ =\ 1)   36.4^20.95^  43.3^15.63^  55.3^16.34^  47.0^10.93^
AlphaNet (_k_\ =\ 2)   41.0^ 0.79^  53.0^ 0.34^  65.9^ 0.19^  56.3^ 0.17^
AlphaNet (_k_\ =\ 3)   42.6^ 1.31^  53.1^ 0.52^  66.1^ 0.29^  56.7^ 0.19^
AlphaNet (_k_\ =\ 4)   43.2^ 1.44^  53.3^ 0.38^  66.3^ 0.28^  56.9^ 0.16^
AlphaNet (_k_\ =\ 5)   43.7^ 1.59^  53.5^ 0.46^  66.4^ 0.29^  57.1^ 0.15^
AlphaNet (_k_\ =\ 6)   42.7^ 1.28^  53.7^ 0.30^  66.6^ 0.22^  57.2^ 0.17^
AlphaNet (_k_\ =\ 7)   43.8^ 1.23^  53.8^ 0.21^  66.5^ 0.25^  57.3^ 0.09^
AlphaNet (_k_\ =\ 8)   43.3^ 1.03^  53.8^ 0.23^  66.6^ 0.16^  57.3^ 0.12^
AlphaNet (_k_\ =\ 9)   44.2^ 1.03^  53.7^ 0.22^  66.5^ 0.20^  57.4^ 0.09^
AlphaNet (_k_\ =\ 10)  43.5^ 1.48^  53.8^ 0.35^  66.7^ 0.26^  57.4^ 0.10^

: Accuracy computed by considering predictions within 3 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_2_semantic3_adjusted_top1}
