Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      40.5         54.8         67.4         57.7
AlphaNet (_k_\ =\ 1)   48.6^19.60^  34.2^14.62^  45.8^15.29^  40.6^10.23^
AlphaNet (_k_\ =\ 2)   46.6^06.77^  47.3^10.21^  60.3^10.39^  52.2^07.98^
AlphaNet (_k_\ =\ 3)   44.6^01.12^  52.2^00.58^  65.4^00.36^  56.3^00.26^
AlphaNet (_k_\ =\ 4)   46.1^00.64^  52.5^00.38^  65.5^00.17^  56.6^00.24^
AlphaNet (_k_\ =\ 5)   45.4^01.45^  52.5^00.44^  65.8^00.27^  56.6^00.20^
AlphaNet (_k_\ =\ 6)   45.9^01.33^  53.0^00.31^  65.9^00.31^  57.0^00.09^
AlphaNet (_k_\ =\ 7)   47.0^01.08^  52.6^00.37^  65.7^00.32^  56.9^00.18^
AlphaNet (_k_\ =\ 8)   46.3^01.01^  53.0^00.33^  65.9^00.23^  57.1^00.13^
AlphaNet (_k_\ =\ 9)   46.1^00.92^  53.0^00.29^  66.1^00.23^  57.1^00.13^
AlphaNet (_k_\ =\ 10)  45.9^01.13^  53.1^00.45^  66.1^00.22^  57.1^00.15^

: Accuracy computed by considering predictions within 3 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_1_semantic3_adjusted_top1}
