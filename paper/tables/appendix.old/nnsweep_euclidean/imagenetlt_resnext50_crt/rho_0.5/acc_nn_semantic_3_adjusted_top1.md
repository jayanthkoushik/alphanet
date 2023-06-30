Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      40.5         54.8         67.4         57.7
AlphaNet (_k_\ =\ 1)   32.3^19.50^  46.3^14.57^  58.4^15.43^  49.1^10.27^
AlphaNet (_k_\ =\ 2)   50.8^07.03^  43.7^11.42^  56.4^12.00^  49.6^09.14^
AlphaNet (_k_\ =\ 3)   47.9^01.71^  50.5^00.54^  64.0^00.41^  55.3^00.34^
AlphaNet (_k_\ =\ 4)   48.2^01.89^  51.1^00.88^  64.4^00.49^  55.8^00.53^
AlphaNet (_k_\ =\ 5)   48.4^01.14^  51.9^00.41^  65.0^00.30^  56.5^00.16^
AlphaNet (_k_\ =\ 6)   48.7^01.31^  51.6^00.48^  64.9^00.36^  56.3^00.22^
AlphaNet (_k_\ =\ 7)   48.8^00.93^  51.8^00.49^  65.0^00.33^  56.5^00.27^
AlphaNet (_k_\ =\ 8)   48.0^01.11^  52.2^00.49^  65.3^00.31^  56.7^00.23^
AlphaNet (_k_\ =\ 9)   49.1^00.62^  52.0^00.29^  65.2^00.26^  56.7^00.18^
AlphaNet (_k_\ =\ 10)  48.4^01.09^  52.3^00.41^  65.5^00.30^  56.8^00.18^

: Accuracy computed by considering predictions within 3 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.5_semantic3_adjusted_top1}
