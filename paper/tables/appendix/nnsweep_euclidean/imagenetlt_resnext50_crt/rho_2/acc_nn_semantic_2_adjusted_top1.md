Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      37.5         51.9         65.7         55.2
AlphaNet (_k_\ =\ 1)   33.6^22.21^  39.9^16.27^  52.4^17.84^  43.9^11.64^
AlphaNet (_k_\ =\ 2)   38.9^01.49^  49.9^00.54^  63.7^00.34^  53.7^00.18^
AlphaNet (_k_\ =\ 3)   39.7^01.25^  50.3^00.29^  64.1^00.33^  54.1^00.14^
AlphaNet (_k_\ =\ 4)   40.3^01.55^  50.4^00.40^  64.4^00.30^  54.4^00.20^
AlphaNet (_k_\ =\ 5)   41.9^01.21^  50.2^00.34^  64.3^00.28^  54.5^00.16^
AlphaNet (_k_\ =\ 6)   40.8^01.22^  50.7^00.29^  64.5^00.19^  54.7^00.08^
AlphaNet (_k_\ =\ 7)   41.2^01.09^  50.7^00.28^  64.6^00.22^  54.8^00.14^
AlphaNet (_k_\ =\ 8)   41.2^01.38^  50.8^00.35^  64.7^00.25^  54.8^00.11^
AlphaNet (_k_\ =\ 9)   41.3^00.94^  50.8^00.25^  64.6^00.17^  54.8^00.07^
AlphaNet (_k_\ =\ 10)  41.4^01.38^  50.8^00.36^  64.7^00.33^  54.9^00.12^

: Accuracy computed by considering predictions within 2 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_2_semantic2_adjusted_top1}
