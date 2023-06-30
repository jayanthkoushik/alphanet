Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      54.7         64.6         74.1        66.9
AlphaNet (_k_\ =\ 1)   48.6^13.17^  58.6^10.24^  67.7^11.01^  60.7^7.35^
AlphaNet (_k_\ =\ 2)   61.2^03.50^  57.8^06.24^  67.5^07.00^  62.0^5.21^
AlphaNet (_k_\ =\ 3)   61.5^02.65^  58.8^04.94^  68.7^05.23^  63.0^4.03^
AlphaNet (_k_\ =\ 4)   60.8^01.02^  61.0^00.54^  71.0^00.43^  64.8^0.43^
AlphaNet (_k_\ =\ 5)   61.1^01.31^  61.6^00.50^  71.5^00.53^  65.3^0.29^
AlphaNet (_k_\ =\ 6)   61.9^00.80^  61.2^00.54^  71.2^00.42^  65.2^0.32^
AlphaNet (_k_\ =\ 7)   61.4^00.84^  61.7^00.49^  71.6^00.48^  65.5^0.34^
AlphaNet (_k_\ =\ 8)   61.5^01.03^  61.5^00.54^  71.6^00.48^  65.4^0.32^
AlphaNet (_k_\ =\ 9)   61.4^00.59^  61.8^00.44^  71.8^00.35^  65.6^0.29^
AlphaNet (_k_\ =\ 10)  61.6^00.67^  61.4^00.34^  71.6^00.31^  65.4^0.24^

: Accuracy computed by considering predictions within 5 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.25_semantic5_adjusted_top1}
