Method                            Few         Med.         Many      Overall
-------------------       -----------  -----------  -----------  -----------
**ImageNet-LT**
Cross entropy                     7.7         37.5         65.9         44.4
NCM                              28.1         45.3         56.6         47.3
$\tau$-normalized                30.7         46.9         59.1         49.4
<!--  -->
cRT                              27.4         46.2         61.8         49.6
$\alpha$-cRT ($\rho=0.5$)  39.7^1.42^   42.0^0.66^   58.3^0.52^   48.0^0.37^
$\alpha$-cRT ($\rho=1$)    34.6^1.88^   43.7^0.51^   59.7^0.43^   48.6^0.24^
$\alpha$-cRT ($\rho=1.5$)  32.6^2.46^   44.4^0.49^   60.3^0.38^   48.9^0.19^
<!--  -->
LWS                              30.4         47.2         60.2         49.9
$\alpha$-LWS ($\rho=0.5$)  46.9^0.98^   38.6^0.87^   52.9^0.86^   45.3^0.69^
$\alpha$-LWS ($\rho=1$)    41.6^1.61^   42.2^0.53^   56.0^0.32^   47.4^0.30^
$\alpha$-LWS ($\rho=1.5$)  40.1^1.99^   43.2^0.98^   56.9^0.76^   48.0^0.53^
<!--  -->
<!--  -->
**Places-LT**
Cross entropy                    8.2         27.3         45.7         30.2
NCM                             27.3         37.1         40.4         36.4
$\tau$-normalized               31.8         40.7         37.8         37.9
<!--  -->
cRT                             24.9         37.6         42.0         36.7
$\alpha$-cRT ($\rho=0.5$) 31.0^0.88^   34.5^0.17^   40.4^0.29^   35.9^0.09^
$\alpha$-cRT ($\rho=1$)   27.0^1.02^   36.1^0.31^   41.3^0.13^   36.2^0.10^
$\alpha$-cRT ($\rho=1.5$) 25.5^0.89^   36.5^0.36^   41.6^0.21^   36.2^0.11^
<!--  -->
LWS                             28.7         39.1         40.6         37.6
$\alpha$-LWS ($\rho=0.5$) 37.1^1.39^   34.4^0.80^   37.7^0.52^   36.1^0.31^
$\alpha$-LWS ($\rho=1$)   34.6^0.97^   35.8^0.54^   38.6^0.39^   36.6^0.22^
$\alpha$-LWS ($\rho=1.5$) 32.2^1.17^   37.2^0.36^   39.5^0.39^   37.0^0.11^

: Mean split accuracy (standard deviation in super-script) of AlphaNet and various baseline methods on ImageNet-LT and Places-LT. $\alpha$\ac{cRT} \aarti{CRT or cRT - be consistent} and $\alpha$\ac{LWS} are AlphaNet models applied over \ac{cRT} and \ac{LWS} features respectively. {#tbl:baselines_imagenetlt_placeslt}
