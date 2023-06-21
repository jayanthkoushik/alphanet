Experiment                    Few        Med.        Many     Overall
---------------------  ----------  ----------  ----------  ----------
Baseline                     85.9        88.7        91.0        89.2
AlphaNet (_k_\ =\ 1)   84.8^3.84^  85.6^3.18^  87.4^4.04^  86.2^2.56^
AlphaNet (_k_\ =\ 2)   86.7^0.78^  87.5^1.36^  89.5^1.74^  88.2^1.22^
AlphaNet (_k_\ =\ 3)   86.8^0.28^  88.1^0.20^  90.1^0.12^  88.7^0.13^
AlphaNet (_k_\ =\ 4)   87.0^0.20^  88.1^0.09^  90.2^0.12^  88.8^0.08^
AlphaNet (_k_\ =\ 5)   87.1^0.32^  88.2^0.09^  90.3^0.18^  88.8^0.09^
AlphaNet (_k_\ =\ 6)   87.2^0.21^  88.2^0.08^  90.3^0.12^  88.9^0.07^
AlphaNet (_k_\ =\ 7)   87.4^0.21^  88.2^0.06^  90.3^0.11^  88.9^0.07^
AlphaNet (_k_\ =\ 8)   87.4^0.13^  88.2^0.08^  90.3^0.16^  88.9^0.08^
AlphaNet (_k_\ =\ 9)   87.3^0.20^  88.3^0.07^  90.4^0.07^  89.0^0.06^
AlphaNet (_k_\ =\ 10)  87.4^0.20^  88.3^0.07^  90.4^0.06^  89.0^0.03^

: Accuracy computed by considering predictions within 10 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.5_semantic10_adjusted_top1}
