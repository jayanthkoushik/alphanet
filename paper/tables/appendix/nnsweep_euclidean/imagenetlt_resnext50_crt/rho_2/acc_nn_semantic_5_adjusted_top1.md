Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      54.7         64.6         74.1        66.9
AlphaNet (_k_\ =\ 1)   51.3^13.94^  56.5^10.90^  65.4^11.74^  59.2^7.84^
AlphaNet (_k_\ =\ 2)   55.0^ 0.85^  63.1^ 0.39^  72.8^ 0.33^  65.7^0.21^
AlphaNet (_k_\ =\ 3)   55.5^ 0.61^  63.4^ 0.29^  73.1^ 0.25^  66.1^0.15^
AlphaNet (_k_\ =\ 4)   56.2^ 1.01^  63.5^ 0.27^  73.2^ 0.30^  66.3^0.19^
AlphaNet (_k_\ =\ 5)   57.1^ 0.85^  63.4^ 0.22^  73.2^ 0.18^  66.3^0.14^
AlphaNet (_k_\ =\ 6)   56.4^ 0.85^  63.8^ 0.22^  73.4^ 0.12^  66.5^0.10^
AlphaNet (_k_\ =\ 7)   56.6^ 0.53^  63.8^ 0.20^  73.4^ 0.13^  66.5^0.12^
AlphaNet (_k_\ =\ 8)   56.7^ 0.73^  63.8^ 0.27^  73.5^ 0.17^  66.6^0.12^
AlphaNet (_k_\ =\ 9)   56.8^ 0.59^  63.9^ 0.20^  73.5^ 0.10^  66.6^0.07^
AlphaNet (_k_\ =\ 10)  56.9^ 0.77^  63.8^ 0.25^  73.5^ 0.25^  66.6^0.12^

: Accuracy computed by considering predictions within 5 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on Euclidean distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_euclidean_imagenetlt_resnext50_crt_rho_2_semantic5_adjusted_top1}
