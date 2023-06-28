Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      37.5         51.9         65.7         55.2
AlphaNet (_k_\ =\ 1)   33.6^22.36^  39.8^16.38^  52.6^17.60^  43.9^11.58^
AlphaNet (_k_\ =\ 2)   38.3^00.89^  50.0^00.32^  64.0^00.24^  53.8^00.15^
AlphaNet (_k_\ =\ 3)   40.0^01.47^  50.1^00.54^  64.1^00.38^  54.1^00.21^
AlphaNet (_k_\ =\ 4)   40.6^01.55^  50.3^00.41^  64.4^00.32^  54.4^00.18^
AlphaNet (_k_\ =\ 5)   41.2^01.73^  50.5^00.47^  64.4^00.33^  54.6^00.16^
AlphaNet (_k_\ =\ 6)   40.1^01.36^  50.8^00.33^  64.7^00.26^  54.7^00.18^
AlphaNet (_k_\ =\ 7)   41.3^01.32^  50.8^00.24^  64.6^00.29^  54.8^00.10^
AlphaNet (_k_\ =\ 8)   40.7^01.11^  50.8^00.24^  64.7^00.18^  54.8^00.12^
AlphaNet (_k_\ =\ 9)   41.7^01.11^  50.8^00.22^  64.6^00.22^  54.9^00.10^
AlphaNet (_k_\ =\ 10)  41.0^01.61^  50.9^00.34^  64.8^00.31^  54.9^00.09^

: Accuracy computed by considering predictions within 2 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_2_semantic2_adjusted_top1}
