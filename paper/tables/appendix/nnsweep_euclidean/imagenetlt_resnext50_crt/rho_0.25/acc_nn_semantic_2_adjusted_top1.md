Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      37.5         51.9         65.7         55.2
AlphaNet (_k_\ =\ 1)   37.9^22.61^  36.7^16.52^  48.9^18.09^  41.5^11.80^
AlphaNet (_k_\ =\ 2)   52.1^ 5.94^  38.3^10.85^  51.7^11.97^  45.3^ 8.99^
AlphaNet (_k_\ =\ 3)   48.9^ 1.91^  46.0^ 0.58^  60.5^ 0.39^  52.0^ 0.36^
AlphaNet (_k_\ =\ 4)   48.6^ 1.78^  46.6^ 0.86^  61.2^ 0.60^  52.5^ 0.45^
AlphaNet (_k_\ =\ 5)   49.3^ 1.17^  47.0^ 0.61^  61.4^ 0.45^  52.8^ 0.41^
AlphaNet (_k_\ =\ 6)   49.1^ 1.36^  47.4^ 0.97^  61.6^ 0.66^  53.1^ 0.53^
AlphaNet (_k_\ =\ 7)   49.6^ 1.50^  47.2^ 0.74^  61.6^ 0.57^  53.1^ 0.42^
AlphaNet (_k_\ =\ 8)   49.7^ 1.09^  47.4^ 0.49^  61.7^ 0.47^  53.2^ 0.31^
AlphaNet (_k_\ =\ 9)   49.0^ 1.71^  47.7^ 0.74^  62.0^ 0.64^  53.4^ 0.39^
AlphaNet (_k_\ =\ 10)  49.2^ 0.73^  47.9^ 0.47^  62.2^ 0.36^  53.6^ 0.29^

: Accuracy computed by considering predictions within 2 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.25_semantic2_adjusted_top1}
