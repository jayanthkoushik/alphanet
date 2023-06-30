Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      54.7         64.6         74.1        66.9
AlphaNet (_k_\ =\ 1)   54.0^14.23^  54.3^11.07^  63.1^11.90^  57.6^7.95^
AlphaNet (_k_\ =\ 2)   62.6^03.78^  55.7^06.91^  65.1^07.67^  60.3^5.75^
AlphaNet (_k_\ =\ 3)   61.0^00.93^  60.6^00.44^  70.6^00.39^  64.5^0.27^
AlphaNet (_k_\ =\ 4)   60.8^01.26^  61.1^00.52^  71.2^00.42^  64.9^0.27^
AlphaNet (_k_\ =\ 5)   61.5^00.87^  61.4^00.43^  71.3^00.38^  65.2^0.30^
AlphaNet (_k_\ =\ 6)   61.2^00.79^  61.7^00.56^  71.5^00.38^  65.4^0.31^
AlphaNet (_k_\ =\ 7)   61.7^00.88^  61.5^00.61^  71.5^00.40^  65.3^0.36^
AlphaNet (_k_\ =\ 8)   61.5^00.55^  61.6^00.29^  71.6^00.29^  65.5^0.19^
AlphaNet (_k_\ =\ 9)   61.2^00.96^  61.9^00.45^  71.7^00.42^  65.6^0.26^
AlphaNet (_k_\ =\ 10)  61.3^00.38^  62.0^00.36^  71.9^00.23^  65.7^0.21^

: Accuracy computed by considering predictions within 5 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.25_semantic5_adjusted_top1}
