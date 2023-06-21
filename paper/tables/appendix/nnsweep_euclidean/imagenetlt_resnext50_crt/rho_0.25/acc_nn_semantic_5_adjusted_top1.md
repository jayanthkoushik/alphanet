Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      54.7         64.6         74.1        66.9
AlphaNet (_k_\ =\ 1)   54.0^14.23^  54.3^11.07^  63.1^11.90^  57.6^7.95^
AlphaNet (_k_\ =\ 2)   62.6^ 3.78^  55.7^ 6.91^  65.1^ 7.67^  60.3^5.75^
AlphaNet (_k_\ =\ 3)   61.0^ 0.93^  60.6^ 0.44^  70.6^ 0.39^  64.5^0.27^
AlphaNet (_k_\ =\ 4)   60.8^ 1.26^  61.1^ 0.52^  71.2^ 0.42^  64.9^0.27^
AlphaNet (_k_\ =\ 5)   61.5^ 0.87^  61.4^ 0.43^  71.3^ 0.38^  65.2^0.30^
AlphaNet (_k_\ =\ 6)   61.2^ 0.79^  61.7^ 0.56^  71.5^ 0.38^  65.4^0.31^
AlphaNet (_k_\ =\ 7)   61.7^ 0.88^  61.5^ 0.61^  71.5^ 0.40^  65.3^0.36^
AlphaNet (_k_\ =\ 8)   61.5^ 0.55^  61.6^ 0.29^  71.6^ 0.29^  65.5^0.19^
AlphaNet (_k_\ =\ 9)   61.2^ 0.96^  61.9^ 0.45^  71.7^ 0.42^  65.6^0.26^
AlphaNet (_k_\ =\ 10)  61.3^ 0.38^  62.0^ 0.36^  71.9^ 0.23^  65.7^0.21^

: Accuracy computed by considering predictions within 5 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.25_semantic5_adjusted_top1}
