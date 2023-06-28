Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      54.7         64.6         74.1        66.9
AlphaNet (_k_\ =\ 1)   59.5^13.17^  50.1^10.24^  58.6^11.01^  54.6^7.35^
AlphaNet (_k_\ =\ 2)   58.3^04.25^  59.4^06.83^  69.1^07.20^  63.0^5.47^
AlphaNet (_k_\ =\ 3)   57.2^00.74^  62.8^00.45^  72.6^00.27^  65.8^0.22^
AlphaNet (_k_\ =\ 4)   58.1^00.51^  63.0^00.33^  72.6^00.17^  66.0^0.19^
AlphaNet (_k_\ =\ 5)   57.8^00.96^  62.9^00.34^  72.8^00.20^  66.0^0.17^
AlphaNet (_k_\ =\ 6)   58.2^00.81^  63.3^00.20^  73.0^00.20^  66.3^0.07^
AlphaNet (_k_\ =\ 7)   58.9^00.66^  63.1^00.23^  72.9^00.28^  66.3^0.16^
AlphaNet (_k_\ =\ 8)   58.3^00.73^  63.3^00.23^  73.1^00.16^  66.4^0.12^
AlphaNet (_k_\ =\ 9)   58.0^00.55^  63.4^00.24^  73.1^00.19^  66.4^0.12^
AlphaNet (_k_\ =\ 10)  58.1^00.77^  63.4^00.30^  73.2^00.17^  66.4^0.11^

: Accuracy computed by considering predictions within 5 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_1_semantic5_adjusted_top1}
