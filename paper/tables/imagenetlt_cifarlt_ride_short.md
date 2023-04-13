Method                             Few         Med.         Many      Overall
-------------------        -----------  -----------  -----------  -----------
**ImageNet-LT**
RIDE                              36.5         54.4         68.9         57.5
$\alpha$-RIDE ($\rho=0.5$)  43.5^0.75^   52.3^0.26^   67.3^0.17^   56.9^0.11^
$\alpha$-RIDE ($\rho=1$)    40.8^1.00^   53.1^0.21^   67.9^0.18^   57.1^0.11^
$\alpha$-RIDE ($\rho=1.5$)  38.2^1.22^   53.6^0.25^   68.4^0.17^   57.2^0.06^
<!--  -->
**CIFAR-100-LT**
RIDE                              25.8         52.1         69.3         50.2
$\alpha$-RIDE ($\rho=0.5$)  30.9^1.82^   47.0^1.25^   65.7^0.94^   48.7^0.41^
$\alpha$-RIDE ($\rho=1$)    27.2^1.69^   49.1^0.85^   67.4^0.62^   48.9^0.30^
$\alpha$-RIDE ($\rho=1.5$)  25.0^1.15^   50.1^1.12^   67.9^1.01^   48.8^0.63^

: Mean split accuracy (standard deviation in super-script) on ImageNet-LT and CIFAR-100-LT using the ensemble \ac{RIDE} model\ [@2020.Yu.Wang]. $\alpha$RIDE applies AlphaNet on average features from the ensemble. {#tbl:ride_imagenetlt_cifarlt_short}
