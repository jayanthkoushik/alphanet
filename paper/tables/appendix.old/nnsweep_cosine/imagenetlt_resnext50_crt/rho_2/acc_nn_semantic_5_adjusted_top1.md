Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      54.7         64.6         74.1        66.9
AlphaNet (_k_\ =\ 1)   51.3^14.08^  56.5^10.95^  65.4^11.77^  59.2^7.86^
AlphaNet (_k_\ =\ 2)   54.8^00.48^  63.2^00.16^  73.0^00.14^  65.9^0.11^
AlphaNet (_k_\ =\ 3)   56.0^00.85^  63.3^00.41^  73.1^00.24^  66.1^0.19^
AlphaNet (_k_\ =\ 4)   56.3^00.89^  63.5^00.31^  73.3^00.24^  66.3^0.16^
AlphaNet (_k_\ =\ 5)   56.6^00.93^  63.7^00.27^  73.4^00.24^  66.5^0.14^
AlphaNet (_k_\ =\ 6)   56.0^00.82^  63.8^00.22^  73.5^00.17^  66.5^0.14^
AlphaNet (_k_\ =\ 7)   56.8^00.69^  63.9^00.15^  73.4^00.16^  66.6^0.06^
AlphaNet (_k_\ =\ 8)   56.5^00.67^  63.9^00.18^  73.5^00.13^  66.6^0.12^
AlphaNet (_k_\ =\ 9)   57.0^00.65^  63.8^00.14^  73.4^00.13^  66.6^0.07^
AlphaNet (_k_\ =\ 10)  56.6^00.92^  63.9^00.20^  73.5^00.18^  66.6^0.07^

: Accuracy computed by considering predictions within 5 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_2_semantic5_adjusted_top1}
