Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      37.5         51.9         65.7         55.2
AlphaNet (_k_\ =\ 1)   42.2^22.21^  33.6^16.27^  45.5^17.84^  39.4^11.64^
AlphaNet (_k_\ =\ 2)   44.7^06.92^  44.3^09.77^  57.9^10.46^  49.6^07.80^
AlphaNet (_k_\ =\ 3)   43.2^01.14^  49.2^00.55^  63.2^00.36^  53.8^00.25^
AlphaNet (_k_\ =\ 4)   43.2^01.58^  49.6^00.50^  63.5^00.41^  54.1^00.29^
AlphaNet (_k_\ =\ 5)   43.5^01.43^  49.8^00.52^  63.8^00.34^  54.3^00.22^
AlphaNet (_k_\ =\ 6)   43.8^00.94^  49.8^00.37^  63.8^00.24^  54.3^00.21^
AlphaNet (_k_\ =\ 7)   44.0^01.04^  49.9^00.40^  63.9^00.26^  54.5^00.17^
AlphaNet (_k_\ =\ 8)   44.4^01.31^  49.8^00.33^  63.8^00.30^  54.5^00.17^
AlphaNet (_k_\ =\ 9)   44.1^01.35^  49.9^00.35^  64.0^00.32^  54.5^00.13^
AlphaNet (_k_\ =\ 10)  43.8^01.63^  50.0^00.50^  64.1^00.46^  54.6^00.21^

: Accuracy computed by considering predictions within 2 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_1_semantic2_adjusted_top1}
