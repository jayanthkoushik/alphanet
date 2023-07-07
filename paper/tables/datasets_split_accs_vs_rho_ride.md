Method                     Few         Med.         Many      Overall
-----------------  -----------  -----------  -----------  -----------
**ImageNet-LT**
\acs{RIDE}                36.5         54.4         68.9         57.5
_α_-\acs{RIDE}
_ρ_\ =\ 0.5         43.5^0.75^   52.3^0.26^   67.3^0.17^   56.9^0.11^
_ρ_\ =\ 1           40.8^1.00^   53.1^0.21^   67.9^0.18^   57.1^0.11^
_ρ_\ =\ 1.5         38.2^1.22^   53.6^0.25^   68.4^0.17^   57.2^0.06^
<!--  -->
**CIFAR-100-LT**
\acs{RIDE}                25.8         52.1         69.3         50.2
_α_-\acs{RIDE}
_ρ_\ =\ 0.5         32.3^1.24^   45.9^0.87^   64.6^0.78^   48.4^0.43^
_ρ_\ =\ 1           27.6^1.41^   49.5^0.83^   67.4^0.70^   49.2^0.16^
_ρ_\ =\ 1.5         25.2^1.11^   50.2^0.57^   68.3^0.34^   49.0^0.26^

: Mean split accuracy in percents (standard deviation in super-script) on ImageNet-LT and CIFAR-100-LT using the ensemble \acs{RIDE} model[@2020.Yu.Wang]. $\alpha$-\acs{RIDE} applies AlphaNet on average features from the ensemble. {#tbl:datasets_split_accs_vs_rho_ride}
