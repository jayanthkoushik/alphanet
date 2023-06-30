Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      37.5         51.9         65.7         55.2
AlphaNet (_k_\ =\ 1)   38.0^22.83^  36.7^16.72^  49.2^17.96^  41.7^11.82^
AlphaNet (_k_\ =\ 2)   46.1^05.34^  45.1^08.04^  58.9^08.67^  50.5^06.47^
AlphaNet (_k_\ =\ 3)   45.7^01.44^  47.7^00.91^  61.9^00.62^  52.9^00.50^
AlphaNet (_k_\ =\ 4)   45.9^01.24^  47.9^00.42^  62.3^00.34^  53.2^00.29^
AlphaNet (_k_\ =\ 5)   46.3^01.85^  48.4^00.58^  62.6^00.51^  53.6^00.28^
AlphaNet (_k_\ =\ 6)   46.4^01.13^  49.0^00.55^  63.0^00.37^  54.0^00.27^
AlphaNet (_k_\ =\ 7)   47.1^00.87^  48.8^00.42^  62.9^00.37^  54.0^00.26^
AlphaNet (_k_\ =\ 8)   47.5^00.95^  48.6^00.52^  62.8^00.38^  53.9^00.31^
AlphaNet (_k_\ =\ 9)   46.8^00.80^  48.9^00.35^  63.1^00.30^  54.1^00.21^
AlphaNet (_k_\ =\ 10)  46.9^00.74^  49.1^00.41^  63.2^00.29^  54.2^00.24^

: Accuracy computed by considering predictions within 2 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.5_semantic2_adjusted_top1}
