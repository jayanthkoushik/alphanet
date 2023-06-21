Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      46.5         59.7         71.0        62.3
AlphaNet (_k_\ =\ 1)   46.0^17.82^  47.2^13.50^  57.4^14.68^  50.9^9.69^
AlphaNet (_k_\ =\ 2)   57.4^ 4.62^  48.5^ 8.82^  59.8^ 9.66^  54.1^7.32^
AlphaNet (_k_\ =\ 3)   55.0^ 1.52^  54.8^ 0.52^  66.8^ 0.44^  59.4^0.32^
AlphaNet (_k_\ =\ 4)   54.6^ 1.43^  55.3^ 0.72^  67.4^ 0.48^  59.9^0.38^
AlphaNet (_k_\ =\ 5)   55.4^ 0.95^  55.7^ 0.52^  67.5^ 0.44^  60.2^0.36^
AlphaNet (_k_\ =\ 6)   55.2^ 0.96^  56.0^ 0.75^  67.8^ 0.47^  60.4^0.41^
AlphaNet (_k_\ =\ 7)   55.6^ 1.17^  55.8^ 0.71^  67.8^ 0.49^  60.4^0.42^
AlphaNet (_k_\ =\ 8)   55.5^ 0.82^  56.0^ 0.37^  67.9^ 0.37^  60.5^0.25^
AlphaNet (_k_\ =\ 9)   55.1^ 1.28^  56.3^ 0.59^  68.0^ 0.55^  60.6^0.33^
AlphaNet (_k_\ =\ 10)  55.2^ 0.49^  56.4^ 0.38^  68.2^ 0.25^  60.8^0.23^

: Accuracy computed by considering predictions within 4 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.25_semantic4_adjusted_top1}
