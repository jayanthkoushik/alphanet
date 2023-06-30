Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      46.5         59.7         71.0        62.3
AlphaNet (_k_\ =\ 1)   42.6^17.49^  49.8^13.29^  60.3^14.49^  52.9^9.57^
AlphaNet (_k_\ =\ 2)   47.2^00.99^  57.9^00.46^  69.5^00.29^  60.9^0.21^
AlphaNet (_k_\ =\ 3)   47.8^00.96^  58.2^00.31^  69.8^00.28^  61.3^0.16^
AlphaNet (_k_\ =\ 4)   48.4^01.28^  58.4^00.32^  70.0^00.28^  61.5^0.20^
AlphaNet (_k_\ =\ 5)   49.5^00.96^  58.3^00.29^  69.9^00.19^  61.6^0.14^
AlphaNet (_k_\ =\ 6)   48.8^00.95^  58.7^00.26^  70.1^00.15^  61.7^0.10^
AlphaNet (_k_\ =\ 7)   49.0^00.83^  58.7^00.27^  70.2^00.18^  61.8^0.15^
AlphaNet (_k_\ =\ 8)   49.1^00.99^  58.7^00.31^  70.3^00.20^  61.8^0.12^
AlphaNet (_k_\ =\ 9)   49.2^00.75^  58.7^00.23^  70.2^00.13^  61.9^0.08^
AlphaNet (_k_\ =\ 10)  49.2^01.04^  58.7^00.33^  70.3^00.28^  61.9^0.14^

: Accuracy computed by considering predictions within 4 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_2_semantic4_adjusted_top1}
