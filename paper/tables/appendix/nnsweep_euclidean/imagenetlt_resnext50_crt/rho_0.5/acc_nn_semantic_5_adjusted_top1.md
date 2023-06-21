Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      54.7         64.6         74.1        66.9
AlphaNet (_k_\ =\ 1)   48.6^13.04^  58.6^10.19^  67.7^10.98^  60.7^7.34^
AlphaNet (_k_\ =\ 2)   60.9^ 4.61^  57.0^ 7.62^  66.4^ 8.32^  61.1^6.23^
AlphaNet (_k_\ =\ 3)   59.3^ 1.07^  61.6^ 0.47^  71.6^ 0.44^  65.1^0.31^
AlphaNet (_k_\ =\ 4)   59.4^ 1.38^  62.0^ 0.63^  71.9^ 0.37^  65.4^0.40^
AlphaNet (_k_\ =\ 5)   59.5^ 0.82^  62.6^ 0.32^  72.4^ 0.25^  65.9^0.17^
AlphaNet (_k_\ =\ 6)   59.6^ 0.89^  62.3^ 0.34^  72.2^ 0.29^  65.8^0.16^
AlphaNet (_k_\ =\ 7)   59.8^ 0.64^  62.6^ 0.36^  72.3^ 0.26^  65.9^0.20^
AlphaNet (_k_\ =\ 8)   59.4^ 0.75^  62.7^ 0.38^  72.5^ 0.23^  66.0^0.19^
AlphaNet (_k_\ =\ 9)   59.9^ 0.47^  62.7^ 0.23^  72.5^ 0.16^  66.1^0.13^
AlphaNet (_k_\ =\ 10)  59.6^ 0.66^  62.9^ 0.31^  72.6^ 0.20^  66.2^0.15^

: Accuracy computed by considering predictions within 5 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.5_semantic5_adjusted_top1}
