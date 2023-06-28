Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      54.7         64.6         74.1        66.9
AlphaNet (_k_\ =\ 1)   56.7^13.94^  52.3^10.90^  60.9^11.74^  56.2^7.84^
AlphaNet (_k_\ =\ 2)   58.4^04.12^  59.5^06.35^  69.2^06.67^  63.1^5.07^
AlphaNet (_k_\ =\ 3)   57.6^00.71^  62.8^00.42^  72.5^00.30^  65.8^0.22^
AlphaNet (_k_\ =\ 4)   57.9^00.86^  63.0^00.34^  72.7^00.31^  66.0^0.22^
AlphaNet (_k_\ =\ 5)   58.0^00.88^  63.2^00.38^  72.9^00.24^  66.2^0.16^
AlphaNet (_k_\ =\ 6)   58.1^00.57^  63.2^00.27^  72.9^00.16^  66.2^0.16^
AlphaNet (_k_\ =\ 7)   58.4^00.65^  63.3^00.26^  72.9^00.15^  66.3^0.10^
AlphaNet (_k_\ =\ 8)   58.5^00.77^  63.2^00.23^  72.9^00.21^  66.3^0.14^
AlphaNet (_k_\ =\ 9)   58.4^00.84^  63.3^00.26^  73.1^00.22^  66.4^0.13^
AlphaNet (_k_\ =\ 10)  58.4^01.00^  63.3^00.32^  73.1^00.26^  66.4^0.13^

: Accuracy computed by considering predictions within 5 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_1_semantic5_adjusted_top1}
