Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      40.5         54.8         67.4         57.7
AlphaNet (_k_\ =\ 1)   32.3^19.60^  46.3^14.62^  58.5^15.29^  49.1^10.23^
AlphaNet (_k_\ =\ 2)   51.5^ 5.12^  44.8^ 9.27^  58.0^ 9.90^  50.8^ 7.56^
AlphaNet (_k_\ =\ 3)   51.7^ 3.92^  46.3^ 7.32^  59.8^ 7.60^  52.2^ 5.92^
AlphaNet (_k_\ =\ 4)   50.7^ 1.61^  49.4^ 0.68^  63.3^ 0.56^  55.0^ 0.48^
AlphaNet (_k_\ =\ 5)   50.7^ 1.95^  50.4^ 0.77^  63.8^ 0.67^  55.6^ 0.39^
AlphaNet (_k_\ =\ 6)   52.1^ 1.42^  49.8^ 0.87^  63.4^ 0.64^  55.4^ 0.46^
AlphaNet (_k_\ =\ 7)   51.2^ 1.11^  50.4^ 0.84^  63.9^ 0.60^  55.7^ 0.49^
AlphaNet (_k_\ =\ 8)   51.5^ 1.63^  50.4^ 0.77^  63.9^ 0.68^  55.7^ 0.42^
AlphaNet (_k_\ =\ 9)   51.2^ 0.84^  50.7^ 0.58^  64.1^ 0.52^  55.9^ 0.38^
AlphaNet (_k_\ =\ 10)  51.5^ 0.91^  50.4^ 0.44^  63.9^ 0.36^  55.7^ 0.30^

: Accuracy computed by considering predictions within 3 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.25_semantic3_adjusted_top1}
