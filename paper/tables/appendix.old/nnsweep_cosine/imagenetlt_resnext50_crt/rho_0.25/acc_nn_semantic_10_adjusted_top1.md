Experiment                    Few        Med.        Many     Overall
---------------------  ----------  ----------  ----------  ----------
Baseline                     85.9        88.7        91.0        89.2
AlphaNet (_k_\ =\ 1)   83.4^3.52^  86.9^2.91^  89.0^3.70^  87.2^2.34^
AlphaNet (_k_\ =\ 2)   87.2^0.77^  86.9^1.69^  88.8^2.06^  87.7^1.50^
AlphaNet (_k_\ =\ 3)   87.4^0.56^  87.5^0.97^  89.3^1.43^  88.2^0.94^
AlphaNet (_k_\ =\ 4)   87.4^0.27^  88.0^0.14^  89.9^0.20^  88.7^0.14^
AlphaNet (_k_\ =\ 5)   87.5^0.35^  88.0^0.13^  90.1^0.23^  88.7^0.14^
AlphaNet (_k_\ =\ 6)   87.7^0.27^  88.0^0.17^  89.9^0.24^  88.7^0.16^
AlphaNet (_k_\ =\ 7)   87.6^0.26^  88.0^0.13^  90.1^0.18^  88.8^0.11^
AlphaNet (_k_\ =\ 8)   87.7^0.25^  88.1^0.14^  90.1^0.20^  88.8^0.11^
AlphaNet (_k_\ =\ 9)   87.6^0.22^  88.1^0.11^  90.2^0.10^  88.9^0.05^
AlphaNet (_k_\ =\ 10)  87.7^0.20^  88.1^0.10^  90.2^0.09^  88.8^0.05^

: Accuracy computed by considering predictions within 10 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.25_semantic10_adjusted_top1}
