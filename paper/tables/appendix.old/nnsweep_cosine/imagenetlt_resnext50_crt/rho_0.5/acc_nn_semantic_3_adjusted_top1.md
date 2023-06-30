Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      40.5         54.8         67.4         57.7
AlphaNet (_k_\ =\ 1)   40.4^21.38^  40.2^15.96^  52.2^16.68^  44.9^11.16^
AlphaNet (_k_\ =\ 2)   48.2^05.01^  48.3^07.69^  61.3^08.10^  53.3^06.13^
AlphaNet (_k_\ =\ 3)   47.8^01.30^  50.8^00.91^  64.1^00.50^  55.5^00.47^
AlphaNet (_k_\ =\ 4)   47.9^01.20^  51.0^00.38^  64.5^00.31^  55.8^00.25^
AlphaNet (_k_\ =\ 5)   48.4^01.73^  51.4^00.51^  64.7^00.42^  56.1^00.22^
AlphaNet (_k_\ =\ 6)   48.7^01.10^  52.0^00.53^  65.1^00.38^  56.6^00.26^
AlphaNet (_k_\ =\ 7)   49.2^00.81^  51.8^00.43^  65.0^00.32^  56.5^00.24^
AlphaNet (_k_\ =\ 8)   49.6^00.92^  51.6^00.52^  64.9^00.39^  56.5^00.31^
AlphaNet (_k_\ =\ 9)   48.9^00.73^  52.0^00.30^  65.2^00.24^  56.7^00.17^
AlphaNet (_k_\ =\ 10)  49.0^00.69^  52.1^00.37^  65.2^00.26^  56.8^00.21^

: Accuracy computed by considering predictions within 3 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.5_semantic3_adjusted_top1}
