Experiment                    Few        Med.        Many     Overall
---------------------  ----------  ----------  ----------  ----------
Baseline                     85.9        88.7        91.0        89.2
AlphaNet (_k_\ =\ 1)   83.4^3.45^  87.0^2.68^  89.0^3.64^  87.3^2.22^
AlphaNet (_k_\ =\ 2)   87.1^1.03^  86.8^2.01^  88.5^2.57^  87.5^1.81^
AlphaNet (_k_\ =\ 3)   87.0^0.24^  88.1^0.08^  90.2^0.14^  88.8^0.06^
AlphaNet (_k_\ =\ 4)   87.1^0.41^  88.1^0.13^  90.3^0.11^  88.8^0.11^
AlphaNet (_k_\ =\ 5)   87.1^0.25^  88.3^0.10^  90.4^0.13^  88.9^0.08^
AlphaNet (_k_\ =\ 6)   87.2^0.22^  88.2^0.09^  90.4^0.14^  88.9^0.07^
AlphaNet (_k_\ =\ 7)   87.3^0.23^  88.2^0.10^  90.3^0.08^  88.9^0.07^
AlphaNet (_k_\ =\ 8)   87.1^0.28^  88.3^0.13^  90.4^0.10^  88.9^0.08^
AlphaNet (_k_\ =\ 9)   87.2^0.16^  88.3^0.08^  90.4^0.11^  89.0^0.06^
AlphaNet (_k_\ =\ 10)  87.2^0.25^  88.3^0.09^  90.4^0.09^  89.0^0.07^

: Accuracy computed by considering predictions within 10 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.5_semantic10_adjusted_top1}
