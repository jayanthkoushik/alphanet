Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      40.5         54.8         67.4         57.7
AlphaNet (_k_\ =\ 1)   44.4^20.85^  37.3^15.57^  48.8^16.50^  42.7^10.98^
AlphaNet (_k_\ =\ 2)   47.0^06.47^  47.5^09.38^  60.3^09.75^  52.4^07.40^
AlphaNet (_k_\ =\ 3)   45.5^01.06^  52.2^00.57^  65.2^00.30^  56.3^00.25^
AlphaNet (_k_\ =\ 4)   45.6^01.50^  52.5^00.48^  65.6^00.38^  56.6^00.26^
AlphaNet (_k_\ =\ 5)   45.9^01.33^  52.8^00.51^  65.7^00.33^  56.8^00.21^
AlphaNet (_k_\ =\ 6)   46.1^00.88^  52.8^00.36^  65.8^00.19^  56.9^00.18^
AlphaNet (_k_\ =\ 7)   46.3^00.96^  52.9^00.37^  65.9^00.23^  57.0^00.15^
AlphaNet (_k_\ =\ 8)   46.7^01.21^  52.8^00.33^  65.8^00.27^  57.0^00.15^
AlphaNet (_k_\ =\ 9)   46.4^01.25^  53.0^00.36^  66.0^00.30^  57.1^00.13^
AlphaNet (_k_\ =\ 10)  46.1^01.51^  53.0^00.48^  66.1^00.39^  57.1^00.19^

: Accuracy computed by considering predictions within 3 WordNet nodes as
correct, for AlphaNet using varying number of nearest neighbors (_k_) based on
Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_1_semantic3_adjusted_top1}
