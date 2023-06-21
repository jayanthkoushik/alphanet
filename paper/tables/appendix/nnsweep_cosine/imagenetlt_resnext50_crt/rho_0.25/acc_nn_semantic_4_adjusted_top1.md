Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      46.5         59.7         71.0        62.3
AlphaNet (_k_\ =\ 1)   39.3^16.54^  52.3^12.55^  63.2^13.41^  54.8^8.92^
AlphaNet (_k_\ =\ 2)   55.6^ 4.25^  51.2^ 7.81^  62.7^ 8.46^  56.2^6.42^
AlphaNet (_k_\ =\ 3)   55.6^ 3.32^  52.4^ 6.19^  64.2^ 6.57^  57.4^5.07^
AlphaNet (_k_\ =\ 4)   54.7^ 1.41^  55.1^ 0.62^  67.3^ 0.53^  59.8^0.47^
AlphaNet (_k_\ =\ 5)   54.9^ 1.69^  55.9^ 0.63^  67.7^ 0.62^  60.3^0.35^
AlphaNet (_k_\ =\ 6)   55.9^ 1.20^  55.5^ 0.68^  67.4^ 0.51^  60.1^0.36^
AlphaNet (_k_\ =\ 7)   55.3^ 0.96^  56.0^ 0.67^  67.9^ 0.53^  60.4^0.41^
AlphaNet (_k_\ =\ 8)   55.5^ 1.35^  55.9^ 0.63^  67.9^ 0.63^  60.5^0.37^
AlphaNet (_k_\ =\ 9)   55.3^ 0.77^  56.1^ 0.50^  68.1^ 0.45^  60.6^0.34^
AlphaNet (_k_\ =\ 10)  55.5^ 0.89^  55.8^ 0.36^  67.9^ 0.34^  60.4^0.28^

: Accuracy computed by considering predictions within 4 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.25_semantic4_adjusted_top1}
