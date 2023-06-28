Experiment                     Few         Med.         Many     Overall
---------------------  -----------  -----------  -----------  ----------
Baseline                      54.7         64.6         74.1        66.9
AlphaNet (_k_\ =\ 1)   54.0^14.37^  54.3^11.17^  63.1^12.01^  57.7^8.02^
AlphaNet (_k_\ =\ 2)   59.2^03.37^  60.2^05.04^  69.8^05.65^  63.8^4.13^
AlphaNet (_k_\ =\ 3)   59.0^00.99^  61.9^00.64^  71.7^00.45^  65.3^0.36^
AlphaNet (_k_\ =\ 4)   59.1^00.88^  61.9^00.40^  72.0^00.22^  65.4^0.28^
AlphaNet (_k_\ =\ 5)   59.5^01.12^  62.2^00.36^  72.1^00.27^  65.7^0.20^
AlphaNet (_k_\ =\ 6)   59.7^00.50^  62.6^00.36^  72.3^00.28^  66.0^0.22^
AlphaNet (_k_\ =\ 7)   60.1^00.54^  62.6^00.32^  72.3^00.24^  66.0^0.18^
AlphaNet (_k_\ =\ 8)   60.4^00.60^  62.4^00.33^  72.3^00.29^  66.0^0.22^
AlphaNet (_k_\ =\ 9)   59.9^00.51^  62.7^00.24^  72.5^00.13^  66.1^0.14^
AlphaNet (_k_\ =\ 10)  60.1^00.45^  62.7^00.27^  72.5^00.15^  66.1^0.15^

: Accuracy computed by considering predictions within 5 WordNet nodes as correct, for AlphaNet using varying number of nearest neighbors (_k_) based on cosine distance, with \acs{cRT} baseline on ImageNet-LT. {#tbl:nnsweep_cosine_imagenetlt_resnext50_crt_rho_0.5_semantic5_adjusted_top1}
