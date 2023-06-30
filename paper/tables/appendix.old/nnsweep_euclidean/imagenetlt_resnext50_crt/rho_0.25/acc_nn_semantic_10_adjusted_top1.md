Experiment                    Few        Med.        Many     Overall
---------------------  ----------  ----------  ----------  ----------
Baseline                     85.9        88.7        91.0        89.2
AlphaNet (_k_\ =\ 1)   84.9^3.72^  85.9^2.91^  87.5^3.97^  86.4^2.42^
AlphaNet (_k_\ =\ 2)   87.6^0.78^  86.5^1.73^  88.0^2.23^  87.2^1.59^
AlphaNet (_k_\ =\ 3)   87.4^0.17^  87.7^0.25^  89.7^0.25^  88.4^0.20^
AlphaNet (_k_\ =\ 4)   87.3^0.30^  87.9^0.23^  90.0^0.24^  88.6^0.17^
AlphaNet (_k_\ =\ 5)   87.5^0.27^  88.0^0.17^  90.0^0.24^  88.7^0.15^
AlphaNet (_k_\ =\ 6)   87.5^0.22^  88.1^0.11^  90.1^0.15^  88.8^0.12^
AlphaNet (_k_\ =\ 7)   87.8^0.27^  88.0^0.17^  90.1^0.15^  88.8^0.11^
AlphaNet (_k_\ =\ 8)   87.6^0.16^  88.1^0.08^  90.1^0.12^  88.8^0.08^
AlphaNet (_k_\ =\ 9)   87.6^0.19^  88.1^0.11^  90.1^0.14^  88.8^0.08^
AlphaNet (_k_\ =\ 10)  87.6^0.15^  88.1^0.08^  90.2^0.07^  88.8^0.06^

: Accuracy computed by considering predictions within 10 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.25_semantic10_adjusted_top1}
