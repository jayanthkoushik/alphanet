Experiment                     Few         Med.         Many      Overall
---------------------  -----------  -----------  -----------  -----------
Baseline                      40.5         54.8         67.4         57.7
AlphaNet (_k_\ =\ 1)   32.3^19.60^  46.3^14.62^  58.5^15.29^  49.1^10.23^
AlphaNet (_k_\ =\ 2)   51.5^05.12^  44.8^09.27^  58.0^09.90^  50.8^07.56^
AlphaNet (_k_\ =\ 3)   51.7^03.92^  46.3^07.32^  59.8^07.60^  52.2^05.92^
AlphaNet (_k_\ =\ 4)   50.7^01.61^  49.4^00.68^  63.3^00.56^  55.0^00.48^
AlphaNet (_k_\ =\ 5)   50.7^01.95^  50.4^00.77^  63.8^00.67^  55.6^00.39^
AlphaNet (_k_\ =\ 6)   52.1^01.42^  49.8^00.87^  63.4^00.64^  55.4^00.46^
AlphaNet (_k_\ =\ 7)   51.2^01.11^  50.4^00.84^  63.9^00.60^  55.7^00.49^
AlphaNet (_k_\ =\ 8)   51.5^01.63^  50.4^00.77^  63.9^00.68^  55.7^00.42^
AlphaNet (_k_\ =\ 9)   51.2^00.84^  50.7^00.58^  64.1^00.52^  55.9^00.38^
AlphaNet (_k_\ =\ 10)  51.5^00.91^  50.4^00.44^  63.9^00.36^  55.7^00.30^

: Accuracy computed by considering predictions within 3 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.25_semantic3_adjusted_top1}
