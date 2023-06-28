Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      46.5         59.7         71.0        62.3
AlphaNet (_k_\ =\ 1)   46.0^17.82^  47.2^13.50^  57.4^14.68^  50.9^9.69^
AlphaNet (_k_\ =\ 2)   57.4^04.62^  48.5^08.82^  59.8^09.66^  54.1^7.32^
AlphaNet (_k_\ =\ 3)   55.0^01.52^  54.8^00.52^  66.8^00.44^  59.4^0.32^
AlphaNet (_k_\ =\ 4)   54.6^01.43^  55.3^00.72^  67.4^00.48^  59.9^0.38^
AlphaNet (_k_\ =\ 5)   55.4^00.95^  55.7^00.52^  67.5^00.44^  60.2^0.36^
AlphaNet (_k_\ =\ 6)   55.2^00.96^  56.0^00.75^  67.8^00.47^  60.4^0.41^
AlphaNet (_k_\ =\ 7)   55.6^01.17^  55.8^00.71^  67.8^00.49^  60.4^0.42^
AlphaNet (_k_\ =\ 8)   55.5^00.82^  56.0^00.37^  67.9^00.37^  60.5^0.25^
AlphaNet (_k_\ =\ 9)   55.1^01.28^  56.3^00.59^  68.0^00.55^  60.6^0.33^
AlphaNet (_k_\ =\ 10)  55.2^00.49^  56.4^00.38^  68.2^00.25^  60.8^0.23^

: Accuracy computed by considering predictions within 4 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.25_semantic4_adjusted_top1}
