Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      40.5         54.8         67.4         57.7
AlphaNet (_k_\ =\ 1)   36.4^20.95^  43.3^15.63^  55.3^16.34^  47.0^10.93^
AlphaNet (_k_\ =\ 2)   41.0^00.79^  53.0^00.34^  65.9^00.19^  56.3^00.17^
AlphaNet (_k_\ =\ 3)   42.6^01.31^  53.1^00.52^  66.1^00.29^  56.7^00.19^
AlphaNet (_k_\ =\ 4)   43.2^01.44^  53.3^00.38^  66.3^00.28^  56.9^00.16^
AlphaNet (_k_\ =\ 5)   43.7^01.59^  53.5^00.46^  66.4^00.29^  57.1^00.15^
AlphaNet (_k_\ =\ 6)   42.7^01.28^  53.7^00.30^  66.6^00.22^  57.2^00.17^
AlphaNet (_k_\ =\ 7)   43.8^01.23^  53.8^00.21^  66.5^00.25^  57.3^00.09^
AlphaNet (_k_\ =\ 8)   43.3^01.03^  53.8^00.23^  66.6^00.16^  57.3^00.12^
AlphaNet (_k_\ =\ 9)   44.2^01.03^  53.7^00.22^  66.5^00.20^  57.4^00.09^
AlphaNet (_k_\ =\ 10)  43.5^01.48^  53.8^00.35^  66.7^00.26^  57.4^00.10^

: Accuracy computed by considering predictions within 3 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_2_semantic3_adjusted_top1}
