Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      37.5         51.9         65.7         55.2
AlphaNet (_k_\ =\ 1)   29.3^20.77^  43.1^15.22^  55.9^16.69^  46.1^10.89^
AlphaNet (_k_\ =\ 2)   48.8^07.53^  40.3^11.93^  53.6^12.99^  46.6^09.69^
AlphaNet (_k_\ =\ 3)   45.8^01.72^  47.4^00.54^  61.8^00.43^  52.7^00.37^
AlphaNet (_k_\ =\ 4)   46.0^01.94^  48.0^00.93^  62.3^00.53^  53.2^00.56^
AlphaNet (_k_\ =\ 5)   46.2^01.21^  48.8^00.41^  63.0^00.29^  53.9^00.16^
AlphaNet (_k_\ =\ 6)   46.5^01.46^  48.5^00.47^  62.8^00.38^  53.7^00.21^
AlphaNet (_k_\ =\ 7)   46.6^00.97^  48.7^00.50^  62.9^00.36^  53.9^00.29^
AlphaNet (_k_\ =\ 8)   45.8^01.18^  49.2^00.50^  63.3^00.34^  54.1^00.24^
AlphaNet (_k_\ =\ 9)   46.9^00.67^  49.0^00.31^  63.2^00.28^  54.2^00.19^
AlphaNet (_k_\ =\ 10)  46.2^01.22^  49.2^00.41^  63.4^00.36^  54.3^00.19^

: Accuracy computed by considering predictions within 2 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.5_semantic2_adjusted_top1}
