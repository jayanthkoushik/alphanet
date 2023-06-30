Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      54.7         64.6         74.1        66.9
AlphaNet (_k_\ =\ 1)   51.3^13.94^  56.5^10.90^  65.4^11.74^  59.2^7.84^
AlphaNet (_k_\ =\ 2)   55.0^00.85^  63.1^00.39^  72.8^00.33^  65.7^0.21^
AlphaNet (_k_\ =\ 3)   55.5^00.61^  63.4^00.29^  73.1^00.25^  66.1^0.15^
AlphaNet (_k_\ =\ 4)   56.2^01.01^  63.5^00.27^  73.2^00.30^  66.3^0.19^
AlphaNet (_k_\ =\ 5)   57.1^00.85^  63.4^00.22^  73.2^00.18^  66.3^0.14^
AlphaNet (_k_\ =\ 6)   56.4^00.85^  63.8^00.22^  73.4^00.12^  66.5^0.10^
AlphaNet (_k_\ =\ 7)   56.6^00.53^  63.8^00.20^  73.4^00.13^  66.5^0.12^
AlphaNet (_k_\ =\ 8)   56.7^00.73^  63.8^00.27^  73.5^00.17^  66.6^0.12^
AlphaNet (_k_\ =\ 9)   56.8^00.59^  63.9^00.20^  73.5^00.10^  66.6^0.07^
AlphaNet (_k_\ =\ 10)  56.9^00.77^  63.8^00.25^  73.5^00.25^  66.6^0.12^

: Accuracy computed by considering predictions within 5 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_2_semantic5_adjusted_top1}
