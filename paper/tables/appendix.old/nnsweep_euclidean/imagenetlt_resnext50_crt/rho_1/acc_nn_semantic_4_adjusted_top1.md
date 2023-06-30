Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      46.5         59.7         71.0        62.3
AlphaNet (_k_\ =\ 1)   49.4^17.49^  44.7^13.29^  54.7^14.49^  49.2^9.57^
AlphaNet (_k_\ =\ 2)   51.7^05.33^  53.4^07.92^  64.8^08.38^  57.6^6.32^
AlphaNet (_k_\ =\ 3)   50.5^00.80^  57.3^00.48^  69.0^00.28^  60.9^0.24^
AlphaNet (_k_\ =\ 4)   50.7^01.14^  57.7^00.37^  69.3^00.33^  61.2^0.24^
AlphaNet (_k_\ =\ 5)   50.8^01.11^  57.9^00.45^  69.5^00.28^  61.4^0.19^
AlphaNet (_k_\ =\ 6)   51.0^00.68^  57.9^00.32^  69.5^00.17^  61.4^0.17^
AlphaNet (_k_\ =\ 7)   51.2^00.88^  58.0^00.34^  69.6^00.20^  61.5^0.14^
AlphaNet (_k_\ =\ 8)   51.5^01.04^  58.0^00.29^  69.6^00.24^  61.5^0.15^
AlphaNet (_k_\ =\ 9)   51.3^01.04^  58.1^00.30^  69.7^00.26^  61.6^0.13^
AlphaNet (_k_\ =\ 10)  51.0^01.27^  58.1^00.43^  69.8^00.34^  61.6^0.18^

: Accuracy computed by considering predictions within 4 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_1_semantic4_adjusted_top1}
