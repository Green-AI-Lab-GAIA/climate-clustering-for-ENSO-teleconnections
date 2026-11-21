### A100 t15_s0  x  A100 p30_daily_cbrt

daily labels: NMI = 0.129, Cramer V = 0.247 (0 = independent, 1 = one determines the other)
ONI from monthly predictors (ridge, 10 contiguous CV folds): levels (domain-mean T, P) R2=0.09; T occupancy R2=0.21; P occupancy R2=0.15; T + P occupancy R2=0.33; T + P occupancy + levels R2=0.33
combined regimes: 126 pairs with >= 0.3 % of days (61% of days); FWER-significant pairs: 13; strong El Nino months 40, strong La Nina months 18

| pair (T, P) | days % | r pair | p FWER | r T alone | r P alone | partial r (pair beyond T, P) | p | lift strong EN | lift strong LN |
|---|---|---|---|---|---|---|---|---|---|
| T6 + P6 | 1.4 | +0.34 | 0.002 | +0.36 | +0.26 | +0.09 | 0.072 | 5.3x | 0.7x |
| T0 + P16 | 0.5 | +0.30 | 0.002 | +0.42 | +0.09 | +0.03 | 0.547 | 5.2x | 0.0x |
| T0 + P15 | 0.6 | +0.27 | 0.002 | +0.42 | +0.00 | +0.04 | 0.390 | 5.7x | 0.0x |
| T0 + P14 | 0.4 | +0.25 | 0.002 | +0.42 | +0.05 | -0.02 | 0.696 | 3.8x | 0.0x |
| T6 + P2 | 0.4 | +0.25 | 0.002 | +0.36 | +0.08 | +0.15 | 0.002 | 4.1x | 0.0x |
| T6 + P5 | 0.6 | +0.24 | 0.002 | +0.36 | +0.01 | +0.15 | 0.002 | 6.2x | 0.3x |
| T0 + P25 | 0.8 | +0.23 | 0.002 | +0.42 | +0.13 | -0.07 | 0.183 | 5.0x | 0.0x |
| T6 + P17 | 0.4 | +0.20 | 0.022 | +0.36 | +0.04 | +0.09 | 0.074 | 3.1x | 0.7x |
| T9 + P22 | 0.6 | -0.19 | 0.040 | -0.26 | -0.17 | -0.04 | 0.425 | 0.1x | 2.9x |
| T5 + P29 | 0.5 | -0.19 | 0.040 | -0.21 | -0.15 | -0.04 | 0.384 | 0.2x | 2.0x |
| T3 + P24 | 0.4 | -0.21 | 0.006 | -0.28 | -0.22 | -0.04 | 0.328 | 0.0x | 7.2x |
| T3 + P19 | 0.8 | -0.21 | 0.006 | -0.28 | -0.14 | +0.01 | 0.911 | 0.0x | 3.1x |
| T3 + P22 | 0.6 | -0.22 | 0.002 | -0.28 | -0.17 | +0.01 | 0.905 | 0.0x | 3.7x |

figure: joint_A100_t15_s0_A100_p30_daily_cbrt.png

### A100 t30_s1  x  A100 p30_daily_cbrt

daily labels: NMI = 0.155, Cramer V = 0.205 (0 = independent, 1 = one determines the other)
ONI from monthly predictors (ridge, 10 contiguous CV folds): levels (domain-mean T, P) R2=0.09; T occupancy R2=0.19; P occupancy R2=0.15; T + P occupancy R2=0.29; T + P occupancy + levels R2=0.29
combined regimes: 71 pairs with >= 0.3 % of days (31% of days); FWER-significant pairs: 7; strong El Nino months 40, strong La Nina months 18

| pair (T, P) | days % | r pair | p FWER | r T alone | r P alone | partial r (pair beyond T, P) | p | lift strong EN | lift strong LN |
|---|---|---|---|---|---|---|---|---|---|
| T26 + P6 | 0.9 | +0.35 | 0.002 | +0.43 | +0.26 | +0.05 | 0.207 | 7.1x | 0.9x |
| T26 + P2 | 0.4 | +0.21 | 0.008 | +0.43 | +0.08 | -0.03 | 0.525 | 5.4x | 0.0x |
| T26 + P25 | 0.4 | +0.21 | 0.008 | +0.43 | +0.13 | -0.04 | 0.211 | 6.9x | 0.0x |
| T26 + P5 | 0.4 | +0.20 | 0.024 | +0.43 | +0.01 | -0.01 | 0.775 | 5.3x | 0.0x |
| T26 + P17 | 0.3 | +0.19 | 0.034 | +0.43 | +0.04 | +0.04 | 0.386 | 4.4x | 0.0x |
| T11 + P29 | 0.3 | -0.16 | 0.111 | -0.20 | -0.15 | -0.04 | 0.338 | 0.0x | 1.9x |
| T19 + P22 | 0.4 | -0.19 | 0.026 | -0.24 | -0.17 | -0.01 | 0.811 | 0.0x | 5.2x |
| T19 + P19 | 0.6 | -0.20 | 0.022 | -0.24 | -0.14 | -0.01 | 0.831 | 0.0x | 3.0x |

figure: joint_A100_t30_s1_A100_p30_daily_cbrt.png

