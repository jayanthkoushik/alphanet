Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      54.7         64.6         74.1        66.9
AlphaNet (_k_\ =\ 1)   48.6^13.17^  58.6^10.24^  67.7^11.01^  60.7^7.35^
AlphaNet (_k_\ =\ 2)   61.2^ 3.50^  57.8^ 6.24^  67.5^ 7.00^  62.0^5.21^
AlphaNet (_k_\ =\ 3)   61.5^ 2.65^  58.8^ 4.94^  68.7^ 5.23^  63.0^4.03^
AlphaNet (_k_\ =\ 4)   60.8^ 1.02^  61.0^ 0.54^  71.0^ 0.43^  64.8^0.43^
AlphaNet (_k_\ =\ 5)   61.1^ 1.31^  61.6^ 0.50^  71.5^ 0.53^  65.3^0.29^
AlphaNet (_k_\ =\ 6)   61.9^ 0.80^  61.2^ 0.54^  71.2^ 0.42^  65.2^0.32^
AlphaNet (_k_\ =\ 7)   61.4^ 0.84^  61.7^ 0.49^  71.6^ 0.48^  65.5^0.34^
AlphaNet (_k_\ =\ 8)   61.5^ 1.03^  61.5^ 0.54^  71.6^ 0.48^  65.4^0.32^
AlphaNet (_k_\ =\ 9)   61.4^ 0.59^  61.8^ 0.44^  71.8^ 0.35^  65.6^0.29^
AlphaNet (_k_\ =\ 10)  61.6^ 0.67^  61.4^ 0.34^  71.6^ 0.31^  65.4^0.24^

: Accuracy computed by considering predictions within 5 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.25_semantic5_adjusted_top1}
