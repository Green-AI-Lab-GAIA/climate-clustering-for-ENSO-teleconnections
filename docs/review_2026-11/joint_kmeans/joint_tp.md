### kmeans_T_15  x  kmeans_P_30

daily labels: NMI = 0.155, Cramer V = 0.284 (0 = independent, 1 = one determines the other)
ONI from monthly predictors (ridge, 10 contiguous CV folds): levels (domain-mean T, P) R2=0.09; T occupancy R2=0.15; P occupancy R2=0.21; T + P occupancy R2=0.24; T + P occupancy + levels R2=0.23
combined regimes: 117 pairs with >= 0.3 % of days (70% of days); FWER-significant pairs: 12; strong El Nino months 40, strong La Nina months 18

| pair (T, P) | days % | r pair | p FWER | r T alone | r P alone | partial r (pair beyond T, P) | p | lift strong EN | lift strong LN |
|---|---|---|---|---|---|---|---|---|---|
| T4 + P27 | 0.8 | +0.34 | 0.002 | +0.43 | +0.35 | -0.08 | 0.070 | 5.8x | 1.1x |
| T4 + P3 | 1.3 | +0.30 | 0.002 | +0.43 | +0.24 | -0.08 | 0.129 | 6.0x | 0.1x |
| T4 + P28 | 0.4 | +0.29 | 0.002 | +0.43 | +0.24 | +0.00 | 0.986 | 4.5x | 0.0x |
| T4 + P12 | 0.3 | +0.26 | 0.002 | +0.43 | +0.02 | +0.05 | 0.278 | 6.3x | 0.0x |
| T4 + P22 | 0.4 | +0.26 | 0.002 | +0.43 | +0.02 | -0.04 | 0.145 | 4.7x | 1.5x |
| T4 + P16 | 0.5 | +0.25 | 0.002 | +0.43 | +0.12 | -0.03 | 0.501 | 3.9x | 0.0x |
| T4 + P26 | 0.4 | +0.23 | 0.006 | +0.43 | -0.03 | -0.01 | 0.857 | 5.3x | 0.8x |
| T4 + P24 | 0.4 | +0.22 | 0.026 | +0.43 | +0.04 | +0.02 | 0.767 | 4.1x | 0.8x |
| T4 + P8 | 0.4 | +0.21 | 0.034 | +0.43 | +0.17 | -0.07 | 0.097 | 6.0x | 0.0x |
| T4 + P13 | 0.5 | +0.21 | 0.034 | +0.43 | +0.20 | -0.12 | 0.002 | 4.3x | 0.0x |
| T3 + P14 | 0.7 | -0.19 | 0.076 | -0.12 | -0.20 | -0.07 | 0.125 | 0.1x | 3.3x |
| T6 + P23 | 1.5 | -0.20 | 0.046 | -0.19 | -0.20 | -0.03 | 0.392 | 0.2x | 2.4x |
| T2 + P1 | 0.8 | -0.21 | 0.030 | -0.26 | -0.25 | +0.02 | 0.714 | 0.1x | 3.3x |

figure: joint_kmeans_T_15_kmeans_P_30.png

### kmeans_T_30  x  kmeans_P_30

daily labels: NMI = 0.188, Cramer V = 0.240 (0 = independent, 1 = one determines the other)
ONI from monthly predictors (ridge, 10 contiguous CV folds): levels (domain-mean T, P) R2=0.09; T occupancy R2=0.29; P occupancy R2=0.21; T + P occupancy R2=0.32; T + P occupancy + levels R2=0.30
combined regimes: 98 pairs with >= 0.3 % of days (47% of days); FWER-significant pairs: 6; strong El Nino months 40, strong La Nina months 18

| pair (T, P) | days % | r pair | p FWER | r T alone | r P alone | partial r (pair beyond T, P) | p | lift strong EN | lift strong LN |
|---|---|---|---|---|---|---|---|---|---|
| T2 + P27 | 0.6 | +0.36 | 0.002 | +0.39 | +0.35 | +0.03 | 0.517 | 6.6x | 0.6x |
| T2 + P3 | 0.9 | +0.29 | 0.002 | +0.39 | +0.24 | -0.08 | 0.046 | 6.5x | 0.3x |
| T24 + P3 | 0.4 | +0.26 | 0.002 | +0.36 | +0.24 | +0.02 | 0.678 | 5.7x | 0.6x |
| T24 + P20 | 0.4 | +0.22 | 0.004 | +0.36 | +0.12 | -0.05 | 0.270 | 5.6x | 0.6x |
| T24 + P8 | 0.5 | +0.22 | 0.014 | +0.36 | +0.17 | -0.05 | 0.308 | 4.2x | 0.4x |
| T2 + P26 | 0.3 | +0.21 | 0.016 | +0.39 | -0.03 | -0.05 | 0.163 | 4.7x | 0.6x |
| T17 + P25 | 0.7 | -0.16 | 0.306 | -0.26 | -0.18 | +0.04 | 0.511 | 0.3x | 0.8x |
| T9 + P14 | 0.4 | -0.16 | 0.300 | -0.15 | -0.20 | -0.04 | 0.366 | 0.0x | 4.0x |
| T27 + P23 | 0.8 | -0.17 | 0.167 | -0.10 | -0.20 | -0.07 | 0.091 | 0.3x | 1.7x |

figure: joint_kmeans_T_30_kmeans_P_30.png

