Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      37.5         51.9         65.7         55.2
AlphaNet (_k_\ =\ 1)   29.3^20.92^  43.0^15.33^  56.0^16.46^  46.2^10.83^
AlphaNet (_k_\ =\ 2)   49.7^05.37^  41.5^09.71^  55.3^10.54^  47.9^07.98^
AlphaNet (_k_\ =\ 3)   49.9^04.19^  43.1^07.67^  57.2^08.24^  49.4^06.31^
AlphaNet (_k_\ =\ 4)   48.7^01.65^  46.4^00.68^  61.1^00.61^  52.3^00.51^
AlphaNet (_k_\ =\ 5)   48.8^02.11^  47.3^00.84^  61.5^00.78^  53.0^00.45^
AlphaNet (_k_\ =\ 6)   50.2^01.50^  46.8^00.86^  61.2^00.67^  52.8^00.47^
AlphaNet (_k_\ =\ 7)   49.3^01.23^  47.3^00.86^  61.7^00.66^  53.1^00.52^
AlphaNet (_k_\ =\ 8)   49.6^01.73^  47.2^00.80^  61.7^00.72^  53.1^00.44^
AlphaNet (_k_\ =\ 9)   49.3^00.91^  47.6^00.61^  61.9^00.55^  53.3^00.41^
AlphaNet (_k_\ =\ 10)  49.5^00.98^  47.3^00.45^  61.7^00.36^  53.2^00.30^

: Accuracy computed by considering predictions within 2 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.25_semantic2_adjusted_top1}
