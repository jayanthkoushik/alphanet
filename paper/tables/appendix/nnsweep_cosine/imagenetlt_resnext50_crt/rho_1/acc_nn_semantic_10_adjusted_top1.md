Experiment                    Few        Med.        Many     Overall
---------------------  ----------  ----------  ----------  ----------
Baseline                     85.9        88.7        91.0        89.2
AlphaNet (_k_\ =\ 1)   86.3^3.52^  84.4^2.91^  85.9^3.70^  85.3^2.34^
AlphaNet (_k_\ =\ 2)   86.5^1.00^  87.3^1.74^  89.4^2.22^  88.0^1.57^
AlphaNet (_k_\ =\ 3)   86.3^0.20^  88.4^0.13^  90.5^0.09^  88.9^0.08^
AlphaNet (_k_\ =\ 4)   86.8^0.21^  88.3^0.14^  90.4^0.11^  88.9^0.09^
AlphaNet (_k_\ =\ 5)   86.7^0.20^  88.4^0.10^  90.6^0.12^  89.0^0.08^
AlphaNet (_k_\ =\ 6)   86.8^0.21^  88.4^0.06^  90.6^0.11^  89.0^0.06^
AlphaNet (_k_\ =\ 7)   87.0^0.27^  88.4^0.08^  90.5^0.09^  89.0^0.05^
AlphaNet (_k_\ =\ 8)   86.8^0.31^  88.4^0.04^  90.6^0.09^  89.0^0.05^
AlphaNet (_k_\ =\ 9)   86.8^0.21^  88.4^0.09^  90.6^0.09^  89.1^0.06^
AlphaNet (_k_\ =\ 10)  86.8^0.28^  88.5^0.11^  90.7^0.08^  89.1^0.07^

: Accuracy computed by considering predictions within 10 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_1_semantic10_adjusted_top1}
