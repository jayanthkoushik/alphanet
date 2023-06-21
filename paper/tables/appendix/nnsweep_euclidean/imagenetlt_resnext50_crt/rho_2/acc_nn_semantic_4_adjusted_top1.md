Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      46.5         59.7         71.0        62.3
AlphaNet (_k_\ =\ 1)   42.6^17.49^  49.8^13.29^  60.3^14.49^  52.9^9.57^
AlphaNet (_k_\ =\ 2)   47.2^ 0.99^  57.9^ 0.46^  69.5^ 0.29^  60.9^0.21^
AlphaNet (_k_\ =\ 3)   47.8^ 0.96^  58.2^ 0.31^  69.8^ 0.28^  61.3^0.16^
AlphaNet (_k_\ =\ 4)   48.4^ 1.28^  58.4^ 0.32^  70.0^ 0.28^  61.5^0.20^
AlphaNet (_k_\ =\ 5)   49.5^ 0.96^  58.3^ 0.29^  69.9^ 0.19^  61.6^0.14^
AlphaNet (_k_\ =\ 6)   48.8^ 0.95^  58.7^ 0.26^  70.1^ 0.15^  61.7^0.10^
AlphaNet (_k_\ =\ 7)   49.0^ 0.83^  58.7^ 0.27^  70.2^ 0.18^  61.8^0.15^
AlphaNet (_k_\ =\ 8)   49.1^ 0.99^  58.7^ 0.31^  70.3^ 0.20^  61.8^0.12^
AlphaNet (_k_\ =\ 9)   49.2^ 0.75^  58.7^ 0.23^  70.2^ 0.13^  61.9^0.08^
AlphaNet (_k_\ =\ 10)  49.2^ 1.04^  58.7^ 0.33^  70.3^ 0.28^  61.9^0.14^

: Accuracy computed by considering predictions within 4 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_2_semantic4_adjusted_top1}
