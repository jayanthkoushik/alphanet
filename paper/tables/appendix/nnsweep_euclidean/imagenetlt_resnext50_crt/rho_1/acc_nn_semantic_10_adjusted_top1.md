Experiment                    Few        Med.        Many     Overall
---------------------  ----------  ----------  ----------  ----------
Baseline                     85.9        88.7        91.0        89.2
AlphaNet (_k_\ =\ 1)   85.6^3.69^  85.3^2.86^  86.7^3.90^  85.9^2.37^
AlphaNet (_k_\ =\ 2)   86.5^0.89^  87.3^1.86^  89.3^2.24^  88.0^1.64^
AlphaNet (_k_\ =\ 3)   86.6^0.22^  88.3^0.13^  90.4^0.13^  88.9^0.10^
AlphaNet (_k_\ =\ 4)   86.8^0.16^  88.3^0.15^  90.4^0.14^  88.9^0.12^
AlphaNet (_k_\ =\ 5)   86.8^0.26^  88.4^0.11^  90.6^0.16^  89.0^0.08^
AlphaNet (_k_\ =\ 6)   86.8^0.19^  88.4^0.07^  90.6^0.10^  89.0^0.05^
AlphaNet (_k_\ =\ 7)   87.0^0.22^  88.4^0.09^  90.5^0.09^  89.0^0.05^
AlphaNet (_k_\ =\ 8)   87.0^0.25^  88.4^0.07^  90.5^0.08^  89.0^0.04^
AlphaNet (_k_\ =\ 9)   86.9^0.28^  88.4^0.08^  90.6^0.12^  89.0^0.07^
AlphaNet (_k_\ =\ 10)  86.9^0.30^  88.4^0.08^  90.6^0.08^  89.0^0.06^

: Accuracy computed by considering predictions within 10 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_1_semantic10_adjusted_top1}
