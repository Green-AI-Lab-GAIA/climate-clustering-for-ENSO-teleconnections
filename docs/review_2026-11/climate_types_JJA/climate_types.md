Climate types first, ENSO second (detrended anomalies, no daily-mean removal).
Warm/cold = top/bottom 20% of the daily domain-mean T anomaly (thresholds +0.37 / -0.32 sigma).
freq = days of the type / days in month; r = partial corr with the index | TNA, detrended, rotation FWER p; lift = mean freq in El Nino (index >= 0.5) or La Nina (<= -0.5) months / overall; "+lvl" also controls the month's mean T level. Types ordered by r with ONI (lag 0). Seasons: share of the type's days in DJF/MAM/JJA/SON.

## all days / eof: k-means on 60 EOFs (78% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 0.97, 0.96; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 24 | +0.54 | -0.10 | +0.92 | -0.58 | -18 | 0/0/100/0 | 3.9 | 1.00 | +0.34 (A3) | +0.35 (0.002) | +0.54 (0.002) | +0.11 (0.795) | +0.28 (0.004) | +nan / +nan / +0.35* / +nan | 1.62 | 0.89 |
| A1 | 9 | -0.19 | +0.04 | -0.91 | +0.41 | +12 | 0/0/100/0 | 1.7 | 1.00 | +0.42 (A4) | +0.18 (0.243) | +0.25 (0.040) | +0.15 (0.404) | +0.14 (0.588) | +nan / +nan / +0.18 / +nan | 1.19 | 0.73 |
| A2 | 9 | +0.06 | +0.48 | -0.23 | +0.44 | +14 | 0/0/100/0 | 2.7 | 1.00 | +0.24 (A1) | +0.13 (0.654) | +0.06 (0.982) | +0.18 (0.203) | +0.13 (0.652) | +nan / +nan / +0.13 / +nan | 1.03 | 0.46 |
| A3 | 24 | +0.05 | -0.25 | +0.40 | -0.43 | -13 | 0/0/100/0 | 4.6 | 1.00 | +0.34 (A0) | -0.12 (0.706) | -0.17 (0.354) | -0.14 (0.529) | -0.10 (0.813) | +nan / +nan / -0.12 / +nan | 0.84 | 1.14 |
| A4 | 7 | -0.93 | -0.12 | -1.65 | +0.01 | +0 | 0/0/100/0 | 3.0 | 1.00 | +0.42 (A1) | -0.25 (0.040) | -0.31 (0.006) | +0.01 (1.000) | -0.21 (0.135) | +nan / +nan / -0.25* / +nan | 0.41 | 1.35 |
| A5 | 26 | -0.15 | +0.12 | -0.17 | +0.44 | +13 | 0/0/100/0 | 3.7 | 1.00 | +0.27 (A4) | -0.27 (0.026) | -0.37 (0.002) | -0.10 (0.851) | -0.23 (0.093) | +nan / +nan / -0.27* / +nan | 0.67 | 1.15 |

* = p < 0.05 (FWER over types) in that season

figure: types_all_eof.png

## all days / msn: k-means on MSN a100-t30_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.70, 0.67; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 14 | +0.31 | +0.14 | +0.15 | +0.22 | +7 | 0/0/100/0 | 2.2 | 0.82 | +0.35 (A2) | +0.34 (0.002) | +0.40 (0.002) | +0.19 (0.249) | +0.29 (0.008) | +nan / +nan / +0.34* / +nan | 1.75 | 0.73 |
| A1 | 20 | +0.51 | -0.20 | +1.02 | -0.71 | -22 | 0/0/100/0 | 4.1 | 1.00 | +0.78 (A3) | +0.19 (0.312) | +0.28 (0.014) | +0.00 (1.000) | +0.15 (0.545) | +nan / +nan / +0.19 / +nan | 1.23 | 1.06 |
| A2 | 17 | -0.03 | +0.26 | -0.31 | +0.64 | +20 | 0/0/100/0 | 2.6 | 0.87 | +0.61 (A4) | +0.02 (1.000) | -0.02 (1.000) | +0.09 (0.891) | +0.01 (1.000) | +nan / +nan / +0.02 / +nan | 0.89 | 0.60 |
| A3 | 19 | +0.07 | -0.20 | +0.43 | -0.47 | -14 | 0/0/100/0 | 2.9 | 0.99 | +0.78 (A1) | -0.16 (0.465) | -0.13 (0.692) | -0.19 (0.235) | -0.11 (0.867) | +nan / +nan / -0.16 / +nan | 0.83 | 1.26 |
| A4 | 16 | -0.68 | +0.05 | -1.33 | +0.40 | +13 | 0/0/100/0 | 3.6 | 1.00 | +0.61 (A2) | -0.17 (0.449) | -0.23 (0.123) | +0.10 (0.863) | -0.14 (0.614) | +nan / +nan / -0.17 / +nan | 0.71 | 1.09 |
| A5 | 14 | -0.20 | -0.09 | -0.01 | -0.18 | -5 | 0/0/100/0 | 2.5 | 0.92 | +0.50 (A3) | -0.30 (0.006) | -0.43 (0.002) | -0.15 (0.511) | -0.27 (0.018) | +nan / +nan / -0.30* / +nan | 0.63 | 1.23 |

* = p < 0.05 (FWER over types) in that season

figure: types_all_msn.png

## warm days / eof: k-means on 60 EOFs (74% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 0.97, 0.97; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W0 | 25 | +0.64 | +0.08 | +0.83 | -0.64 | -19 | 0/0/100/0 | 2.3 | 1.00 | +0.40 (W1) | +0.29 (0.012) | +0.28 (0.030) | +0.16 (0.384) | +0.26 (0.056) | +nan / +nan / +0.29* / +nan | 1.91 | 0.53 |
| W1 | 43 | +0.60 | -0.32 | +1.26 | -1.01 | -30 | 0/0/100/0 | 3.1 | 1.00 | +0.59 (W2) | +0.17 (0.392) | +0.21 (0.207) | +0.01 (1.000) | +0.15 (0.501) | +nan / +nan / +0.17 / +nan | 1.39 | 0.98 |
| W2 | 32 | +0.61 | +0.01 | +0.88 | -0.10 | -3 | 0/0/100/0 | 2.4 | 1.00 | +0.59 (W1) | +0.12 (0.638) | +0.41 (0.002) | -0.08 (0.815) | +0.05 (0.962) | +nan / +nan / +0.12 / +nan | 1.75 | 1.25 |

* = p < 0.05 (FWER over types) in that season

figure: types_warm_eof.png

## warm days / msn: k-means on MSN a100-t30_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.61, 0.58; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W0 | 32 | +0.67 | +0.06 | +0.85 | -0.30 | -9 | 0/0/100/0 | 2.4 | 0.97 | +0.92 (W1) | +0.31 (0.002) | +0.47 (0.002) | +0.13 (0.523) | +0.25 (0.046) | +nan / +nan / +0.31* / +nan | 1.98 | 0.93 |
| W1 | 24 | +0.49 | +0.01 | +0.73 | -0.33 | -10 | 0/0/100/0 | 2.1 | 0.96 | +0.92 (W0) | +0.15 (0.439) | +0.23 (0.072) | +0.01 (0.998) | +0.09 (0.791) | +nan / +nan / +0.15 / +nan | 1.99 | 0.62 |
| W2 | 44 | +0.63 | -0.30 | +1.33 | -1.02 | -32 | 0/0/100/0 | 3.0 | 1.00 | +0.70 (W1) | +0.12 (0.662) | +0.26 (0.038) | -0.07 (0.895) | +0.09 (0.783) | +nan / +nan / +0.12 / +nan | 1.19 | 1.15 |

* = p < 0.05 (FWER over types) in that season

figure: types_warm_msn.png

## cold days / eof: k-means on 54 EOFs (80% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 0.91, 0.91; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 33 | -0.55 | +0.35 | -1.42 | +1.11 | +35 | 0/0/100/0 | 1.9 | 1.00 | +0.69 (C1) | -0.04 (0.942) | -0.15 (0.247) | +0.16 (0.123) | -0.01 (1.000) | +nan / +nan / -0.04 / +nan | 0.49 | 0.95 |
| C1 | 22 | -1.12 | -0.18 | -2.11 | -0.01 | -0 | 0/0/100/0 | 2.3 | 1.00 | +0.69 (C0) | -0.18 (0.123) | -0.21 (0.060) | +0.04 (0.948) | -0.16 (0.189) | +nan / +nan / -0.18 / +nan | 0.46 | 1.16 |
| C2 | 45 | -0.53 | -0.12 | -0.58 | +0.00 | +0 | 0/0/100/0 | 2.3 | 1.00 | +0.38 (C1) | -0.24 (0.036) | -0.35 (0.002) | -0.02 (0.992) | -0.16 (0.177) | +nan / +nan / -0.24* / +nan | 0.60 | 1.51 |

* = p < 0.05 (FWER over types) in that season

figure: types_cold_eof.png

## cold days / msn: k-means on MSN a100-t30_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.72, 0.30; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 28 | -0.54 | +0.27 | -1.44 | +1.04 | +33 | 0/0/100/0 | 1.6 | 0.98 | +0.85 (C1) | -0.05 (0.934) | -0.14 (0.360) | +0.12 (0.439) | -0.02 (0.988) | +nan / +nan / -0.05 / +nan | 0.57 | 0.96 |
| C1 | 42 | -0.87 | +0.03 | -1.53 | +0.41 | +13 | 0/0/100/0 | 3.6 | 0.91 | +0.85 (C0) | -0.15 (0.274) | -0.21 (0.066) | +0.09 (0.648) | -0.12 (0.441) | +nan / +nan / -0.15 / +nan | 0.55 | 1.16 |
| C2 | 30 | -0.51 | -0.21 | -0.49 | -0.31 | -10 | 0/0/100/0 | 1.8 | 0.85 | +0.46 (C1) | -0.21 (0.074) | -0.33 (0.010) | -0.05 (0.924) | -0.14 (0.342) | +nan / +nan / -0.21 / +nan | 0.47 | 1.64 |

* = p < 0.05 (FWER over types) in that season

figure: types_cold_msn.png

