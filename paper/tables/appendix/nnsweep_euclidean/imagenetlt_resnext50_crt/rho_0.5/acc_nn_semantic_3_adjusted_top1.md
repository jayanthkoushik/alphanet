Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      40.5         54.8         67.4         57.7
AlphaNet (_k_\ =\ 1)   32.3^19.50^  46.3^14.57^  58.4^15.43^  49.1^10.27^
AlphaNet (_k_\ =\ 2)   50.8^ 7.03^  43.7^11.42^  56.4^12.00^  49.6^ 9.14^
AlphaNet (_k_\ =\ 3)   47.9^ 1.71^  50.5^ 0.54^  64.0^ 0.41^  55.3^ 0.34^
AlphaNet (_k_\ =\ 4)   48.2^ 1.89^  51.1^ 0.88^  64.4^ 0.49^  55.8^ 0.53^
AlphaNet (_k_\ =\ 5)   48.4^ 1.14^  51.9^ 0.41^  65.0^ 0.30^  56.5^ 0.16^
AlphaNet (_k_\ =\ 6)   48.7^ 1.31^  51.6^ 0.48^  64.9^ 0.36^  56.3^ 0.22^
AlphaNet (_k_\ =\ 7)   48.8^ 0.93^  51.8^ 0.49^  65.0^ 0.33^  56.5^ 0.27^
AlphaNet (_k_\ =\ 8)   48.0^ 1.11^  52.2^ 0.49^  65.3^ 0.31^  56.7^ 0.23^
AlphaNet (_k_\ =\ 9)   49.1^ 0.62^  52.0^ 0.29^  65.2^ 0.26^  56.7^ 0.18^
AlphaNet (_k_\ =\ 10)  48.4^ 1.09^  52.3^ 0.41^  65.5^ 0.30^  56.8^ 0.18^

: Accuracy computed by considering predictions within 3 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_0.5_semantic3_adjusted_top1}
