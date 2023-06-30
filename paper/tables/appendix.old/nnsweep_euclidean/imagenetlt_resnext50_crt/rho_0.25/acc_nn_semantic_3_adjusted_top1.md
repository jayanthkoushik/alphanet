Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      40.5         54.8         67.4         57.7
AlphaNet (_k_\ =\ 1)   40.4^21.24^  40.2^15.80^  51.9^16.72^  44.7^11.12^
AlphaNet (_k_\ =\ 2)   53.8^05.61^  41.6^10.43^  54.6^11.14^  48.3^08.52^
AlphaNet (_k_\ =\ 3)   50.9^01.87^  49.1^00.58^  62.7^00.46^  54.6^00.32^
AlphaNet (_k_\ =\ 4)   50.6^01.61^  49.8^00.85^  63.4^00.54^  55.1^00.42^
AlphaNet (_k_\ =\ 5)   51.3^01.09^  50.1^00.60^  63.6^00.44^  55.5^00.39^
AlphaNet (_k_\ =\ 6)   51.1^01.29^  50.5^00.93^  63.8^00.56^  55.7^00.49^
AlphaNet (_k_\ =\ 7)   51.5^01.43^  50.3^00.77^  63.8^00.55^  55.6^00.44^
AlphaNet (_k_\ =\ 8)   51.6^01.00^  50.5^00.45^  63.9^00.46^  55.8^00.29^
AlphaNet (_k_\ =\ 9)   51.1^01.61^  50.8^00.71^  64.1^00.63^  56.0^00.37^
AlphaNet (_k_\ =\ 10)  51.1^00.70^  51.0^00.43^  64.3^00.29^  56.2^00.25^

: Accuracy computed by considering predictions within 3 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.25_semantic3_adjusted_top1}
