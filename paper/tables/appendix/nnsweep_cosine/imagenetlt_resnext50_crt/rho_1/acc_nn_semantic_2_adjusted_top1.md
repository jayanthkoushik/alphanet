Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      37.5         51.9         65.7         55.2
AlphaNet (_k_\ =\ 1)   46.6^20.92^  30.3^15.33^  42.4^16.46^  37.2^10.83^
AlphaNet (_k_\ =\ 2)   44.3^07.25^  44.0^10.76^  57.9^11.24^  49.4^08.50^
AlphaNet (_k_\ =\ 3)   42.3^01.23^  49.3^00.56^  63.4^00.41^  53.8^00.26^
AlphaNet (_k_\ =\ 4)   43.7^00.72^  49.5^00.42^  63.5^00.21^  54.1^00.26^
AlphaNet (_k_\ =\ 5)   43.1^01.59^  49.5^00.46^  63.7^00.31^  54.1^00.20^
AlphaNet (_k_\ =\ 6)   43.6^01.41^  50.0^00.33^  63.9^00.36^  54.5^00.12^
AlphaNet (_k_\ =\ 7)   44.8^01.19^  49.6^00.37^  63.7^00.35^  54.4^00.18^
AlphaNet (_k_\ =\ 8)   44.0^01.09^  50.0^00.34^  63.9^00.25^  54.6^00.14^
AlphaNet (_k_\ =\ 9)   43.7^00.99^  50.1^00.29^  64.1^00.27^  54.6^00.15^
AlphaNet (_k_\ =\ 10)  43.6^01.26^  50.1^00.44^  64.1^00.27^  54.6^00.15^

: Accuracy computed by considering predictions within 2 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_1_semantic2_adjusted_top1}
