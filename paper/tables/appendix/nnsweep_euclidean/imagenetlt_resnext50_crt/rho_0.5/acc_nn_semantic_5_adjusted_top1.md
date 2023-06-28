Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      54.7         64.6         74.1        66.9
AlphaNet (_k_\ =\ 1)   48.6^13.04^  58.6^10.19^  67.7^10.98^  60.7^7.34^
AlphaNet (_k_\ =\ 2)   60.9^04.61^  57.0^07.62^  66.4^08.32^  61.1^6.23^
AlphaNet (_k_\ =\ 3)   59.3^01.07^  61.6^00.47^  71.6^00.44^  65.1^0.31^
AlphaNet (_k_\ =\ 4)   59.4^01.38^  62.0^00.63^  71.9^00.37^  65.4^0.40^
AlphaNet (_k_\ =\ 5)   59.5^00.82^  62.6^00.32^  72.4^00.25^  65.9^0.17^
AlphaNet (_k_\ =\ 6)   59.6^00.89^  62.3^00.34^  72.2^00.29^  65.8^0.16^
AlphaNet (_k_\ =\ 7)   59.8^00.64^  62.6^00.36^  72.3^00.26^  65.9^0.20^
AlphaNet (_k_\ =\ 8)   59.4^00.75^  62.7^00.38^  72.5^00.23^  66.0^0.19^
AlphaNet (_k_\ =\ 9)   59.9^00.47^  62.7^00.23^  72.5^00.16^  66.1^0.13^
AlphaNet (_k_\ =\ 10)  59.6^00.66^  62.9^00.31^  72.6^00.20^  66.2^0.15^

: Accuracy computed by considering predictions within 5 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.5_semantic5_adjusted_top1}
