Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      46.5         59.7         71.0        62.3
AlphaNet (_k_\ =\ 1)   42.7^17.69^  49.7^13.42^  60.5^14.34^  52.9^9.54^
AlphaNet (_k_\ =\ 2)   46.8^00.76^  58.0^00.24^  69.7^00.18^  61.0^0.13^
AlphaNet (_k_\ =\ 3)   48.1^01.14^  58.1^00.44^  69.8^00.31^  61.2^0.20^
AlphaNet (_k_\ =\ 4)   48.5^01.24^  58.3^00.33^  70.0^00.28^  61.5^0.17^
AlphaNet (_k_\ =\ 5)   49.1^01.23^  58.5^00.35^  70.1^00.27^  61.7^0.15^
AlphaNet (_k_\ =\ 6)   48.2^01.08^  58.7^00.26^  70.2^00.19^  61.7^0.16^
AlphaNet (_k_\ =\ 7)   49.2^00.96^  58.8^00.18^  70.2^00.23^  61.9^0.08^
AlphaNet (_k_\ =\ 8)   48.7^00.92^  58.7^00.21^  70.2^00.15^  61.8^0.13^
AlphaNet (_k_\ =\ 9)   49.4^00.88^  58.7^00.20^  70.2^00.17^  61.8^0.10^
AlphaNet (_k_\ =\ 10)  48.9^01.25^  58.8^00.29^  70.3^00.23^  61.9^0.09^

: Accuracy computed by considering predictions within 4 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_2_semantic4_adjusted_top1}
