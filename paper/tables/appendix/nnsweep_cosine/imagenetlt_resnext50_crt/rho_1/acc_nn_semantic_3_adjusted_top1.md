Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      40.5         54.8         67.4         57.7
AlphaNet (_k_\ =\ 1)   48.6^19.60^  34.2^14.62^  45.8^15.29^  40.6^10.23^
AlphaNet (_k_\ =\ 2)   46.6^ 6.77^  47.3^10.21^  60.3^10.39^  52.2^ 7.98^
AlphaNet (_k_\ =\ 3)   44.6^ 1.12^  52.2^ 0.58^  65.4^ 0.36^  56.3^ 0.26^
AlphaNet (_k_\ =\ 4)   46.1^ 0.64^  52.5^ 0.38^  65.5^ 0.17^  56.6^ 0.24^
AlphaNet (_k_\ =\ 5)   45.4^ 1.45^  52.5^ 0.44^  65.8^ 0.27^  56.6^ 0.20^
AlphaNet (_k_\ =\ 6)   45.9^ 1.33^  53.0^ 0.31^  65.9^ 0.31^  57.0^ 0.09^
AlphaNet (_k_\ =\ 7)   47.0^ 1.08^  52.6^ 0.37^  65.7^ 0.32^  56.9^ 0.18^
AlphaNet (_k_\ =\ 8)   46.3^ 1.01^  53.0^ 0.33^  65.9^ 0.23^  57.1^ 0.13^
AlphaNet (_k_\ =\ 9)   46.1^ 0.92^  53.0^ 0.29^  66.1^ 0.23^  57.1^ 0.13^
AlphaNet (_k_\ =\ 10)  45.9^ 1.13^  53.1^ 0.45^  66.1^ 0.22^  57.1^ 0.15^

: Accuracy computed by considering predictions within 3 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_1_semantic3_adjusted_top1}
