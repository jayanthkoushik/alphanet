Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      40.5         54.8         67.4         57.7
AlphaNet (_k_\ =\ 1)   36.4^20.85^  43.3^15.57^  55.2^16.50^  46.9^10.98^
AlphaNet (_k_\ =\ 2)   41.5^01.33^  52.8^00.56^  65.7^00.30^  56.2^00.21^
AlphaNet (_k_\ =\ 3)   42.3^01.21^  53.2^00.33^  66.0^00.33^  56.7^00.14^
AlphaNet (_k_\ =\ 4)   42.8^01.47^  53.4^00.39^  66.3^00.29^  56.9^00.19^
AlphaNet (_k_\ =\ 5)   44.4^01.07^  53.2^00.33^  66.2^00.22^  57.0^00.14^
AlphaNet (_k_\ =\ 6)   43.4^01.09^  53.7^00.30^  66.5^00.16^  57.2^00.09^
AlphaNet (_k_\ =\ 7)   43.7^01.05^  53.7^00.27^  66.5^00.20^  57.3^00.14^
AlphaNet (_k_\ =\ 8)   43.7^01.26^  53.7^00.34^  66.6^00.23^  57.3^00.11^
AlphaNet (_k_\ =\ 9)   43.9^00.89^  53.8^00.26^  66.6^00.15^  57.4^00.08^
AlphaNet (_k_\ =\ 10)  43.9^01.31^  53.8^00.35^  66.6^00.31^  57.4^00.12^

: Accuracy computed by considering predictions within 3 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_2_semantic3_adjusted_top1}
