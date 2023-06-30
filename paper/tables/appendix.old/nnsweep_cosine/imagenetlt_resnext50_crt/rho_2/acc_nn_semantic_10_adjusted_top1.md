Experiment                    Few        Med.        Many     Overall
---------------------  ----------  ----------  ----------  ----------
Baseline                     85.9        88.7        91.0        89.2
AlphaNet (_k_\ =\ 1)   84.1^3.77^  86.3^3.11^  88.2^3.96^  86.7^2.50^
AlphaNet (_k_\ =\ 2)   85.6^0.24^  88.3^0.12^  90.7^0.16^  88.9^0.10^
AlphaNet (_k_\ =\ 3)   86.1^0.27^  88.4^0.13^  90.7^0.08^  89.0^0.06^
AlphaNet (_k_\ =\ 4)   86.3^0.33^  88.4^0.11^  90.7^0.12^  89.0^0.06^
AlphaNet (_k_\ =\ 5)   86.5^0.23^  88.4^0.09^  90.7^0.13^  89.0^0.06^
AlphaNet (_k_\ =\ 6)   86.3^0.27^  88.5^0.05^  90.8^0.08^  89.1^0.04^
AlphaNet (_k_\ =\ 7)   86.5^0.24^  88.5^0.06^  90.7^0.08^  89.1^0.05^
AlphaNet (_k_\ =\ 8)   86.4^0.26^  88.5^0.06^  90.8^0.06^  89.1^0.02^
AlphaNet (_k_\ =\ 9)   86.5^0.20^  88.5^0.03^  90.8^0.06^  89.1^0.02^
AlphaNet (_k_\ =\ 10)  86.4^0.29^  88.5^0.08^  90.8^0.09^  89.1^0.03^

: Accuracy computed by considering predictions within 10 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_2_semantic10_adjusted_top1}
