Experiment                    Few        Med.        Many     Overall
---------------------  ----------  ----------  ----------  ----------
Baseline                     85.9        88.7        91.0        89.2
AlphaNet (_k_\ =\ 1)   84.1^3.69^  86.5^2.86^  88.2^3.90^  86.8^2.37^
AlphaNet (_k_\ =\ 2)   85.8^0.21^  88.3^0.08^  90.6^0.09^  88.8^0.05^
AlphaNet (_k_\ =\ 3)   86.2^0.27^  88.4^0.13^  90.7^0.13^  89.0^0.09^
AlphaNet (_k_\ =\ 4)   86.3^0.32^  88.4^0.09^  90.7^0.13^  89.0^0.08^
AlphaNet (_k_\ =\ 5)   86.5^0.28^  88.4^0.08^  90.7^0.12^  89.1^0.07^
AlphaNet (_k_\ =\ 6)   86.3^0.31^  88.5^0.09^  90.7^0.09^  89.1^0.06^
AlphaNet (_k_\ =\ 7)   86.4^0.21^  88.5^0.05^  90.8^0.06^  89.1^0.03^
AlphaNet (_k_\ =\ 8)   86.4^0.28^  88.5^0.06^  90.8^0.08^  89.1^0.05^
AlphaNet (_k_\ =\ 9)   86.4^0.23^  88.5^0.05^  90.7^0.08^  89.1^0.03^
AlphaNet (_k_\ =\ 10)  86.5^0.25^  88.5^0.07^  90.8^0.10^  89.1^0.05^

: Accuracy computed by considering predictions within 10 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_2_semantic10_adjusted_top1}
