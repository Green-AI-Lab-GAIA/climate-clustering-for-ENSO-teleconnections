Climate types first, ENSO second (detrended anomalies, no daily-mean removal).
Warm/cold = top/bottom 20% of the daily domain-mean T anomaly (thresholds +0.36 / -0.36 sigma).
freq = days of the type / days in month; r = partial corr with the index | TNA, detrended, rotation FWER p; lift = mean freq in El Nino (index >= 0.5) or La Nina (<= -0.5) months / overall; "+lvl" also controls the month's mean T level. Types ordered by r with ONI (lag 0). Seasons: share of the type's days in DJF/MAM/JJA/SON.

## all days / eof: k-means on 60 EOFs (74% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 0.99, 0.99; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 19 | +0.72 | -0.20 | +1.10 | -1.16 | -20 | 24/25/26/25 | 4.7 | 1.00 | +0.42 (A4) | +0.55 (0.002) | +0.65 (0.002) | +0.15 (0.123) | +0.50 (0.002) | +0.71* / +0.54* / +0.39* / +0.50* | 2.07 | 0.43 |
| A1 | 15 | -0.05 | +0.16 | -0.42 | +0.83 | +14 | 23/23/24/30 | 1.9 | 1.00 | +0.29 (A3) | +0.03 (0.980) | +0.08 (0.620) | +0.04 (0.958) | +0.04 (0.954) | -0.05 / +0.18 / +0.22 / -0.07 | 1.01 | 0.88 |
| A2 | 15 | +0.00 | +0.07 | -0.13 | +0.33 | +5 | 33/29/13/25 | 2.5 | 1.00 | -0.02 (A0) | -0.01 (1.000) | -0.07 (0.773) | +0.06 (0.773) | +0.01 (1.000) | +0.00 / -0.00 / +0.13 / -0.12 | 0.94 | 0.90 |
| A3 | 5 | -0.77 | -0.13 | -1.38 | +0.20 | +4 | 7/29/40/24 | 2.6 | 1.00 | +0.44 (A5) | -0.12 (0.348) | -0.12 (0.306) | +0.14 (0.155) | -0.12 (0.344) | -0.15 / -0.17 / -0.17 / -0.13 | 0.60 | 0.96 |
| A4 | 26 | +0.09 | -0.13 | +0.41 | -0.64 | -12 | 21/24/32/23 | 3.2 | 1.00 | +0.42 (A0) | -0.18 (0.062) | -0.19 (0.052) | -0.18 (0.040) | -0.16 (0.097) | -0.16 / -0.10 / -0.29* / -0.24 | 0.75 | 1.07 |
| A5 | 20 | -0.41 | +0.16 | -0.57 | +0.76 | +12 | 30/26/21/23 | 3.3 | 1.00 | +0.44 (A3) | -0.43 (0.002) | -0.52 (0.002) | -0.05 (0.847) | -0.41 (0.002) | -0.62* / -0.54* / -0.27* / -0.20 | 0.47 | 1.62 |

* = p < 0.05 (FWER over types) in that season

figure: types_all_eof.png

## all days / msn: k-means on MSN a100-t30_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.67, 0.64; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 19 | +0.65 | -0.16 | +0.97 | -0.92 | -16 | 21/25/28/27 | 4.8 | 0.99 | +0.78 (A2) | +0.51 (0.002) | +0.61 (0.002) | +0.11 (0.543) | +0.45 (0.002) | +0.67* / +0.54* / +0.31* / +0.46* | 1.90 | 0.53 |
| A1 | 16 | +0.14 | +0.15 | -0.12 | +0.73 | +12 | 28/25/25/23 | 2.4 | 0.92 | +0.75 (A3) | +0.19 (0.074) | +0.23 (0.020) | +0.10 (0.596) | +0.18 (0.070) | +0.23 / +0.27 / +0.33* / -0.01 | 1.32 | 0.69 |
| A2 | 18 | +0.16 | -0.19 | +0.53 | -0.92 | -17 | 19/24/31/25 | 2.4 | 0.98 | +0.78 (A0) | -0.11 (0.569) | -0.10 (0.664) | -0.16 (0.119) | -0.08 (0.819) | -0.03 / -0.02 / -0.13 / -0.24 | 0.85 | 1.05 |
| A3 | 18 | -0.22 | +0.19 | -0.57 | +1.04 | +18 | 23/25/25/27 | 2.3 | 0.98 | +0.79 (A4) | -0.20 (0.058) | -0.24 (0.012) | +0.01 (1.000) | -0.18 (0.093) | -0.28 / -0.20 / -0.12 / -0.20 | 0.65 | 1.07 |
| A4 | 14 | -0.60 | +0.07 | -0.88 | +0.55 | +9 | 26/23/23/28 | 3.7 | 0.99 | +0.79 (A3) | -0.30 (0.002) | -0.36 (0.002) | +0.04 (0.994) | -0.30 (0.002) | -0.46* / -0.42* / -0.21 / -0.08 | 0.51 | 1.39 |
| A5 | 15 | -0.18 | -0.10 | +0.05 | -0.59 | -9 | 33/30/18/19 | 2.3 | 0.98 | +0.71 (A2) | -0.31 (0.002) | -0.41 (0.002) | -0.09 (0.720) | -0.27 (0.002) | -0.42* / -0.44* / -0.31* / -0.08 | 0.61 | 1.40 |

* = p < 0.05 (FWER over types) in that season

figure: types_all_msn.png

## warm days / eof: k-means on 60 EOFs (71% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 1.00, 1.00; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W0 | 22 | +1.05 | -0.29 | +1.52 | -1.65 | -25 | 32/38/10/21 | 4.0 | 1.00 | +0.93 (W2) | +0.44 (0.002) | +0.54 (0.002) | +0.12 (0.119) | +0.41 (0.002) | +0.59* / +0.41* / +0.21 / +0.46* | 2.79 | 0.17 |
| W1 | 35 | +0.63 | -0.13 | +0.86 | -0.73 | -13 | 24/23/27/26 | 2.6 | 1.00 | +0.55 (W0) | +0.39 (0.002) | +0.40 (0.002) | +0.11 (0.161) | +0.35 (0.002) | +0.50* / +0.34* / +0.37* / +0.34* | 2.02 | 0.33 |
| W2 | 44 | +0.53 | -0.17 | +0.97 | -0.94 | -17 | 20/23/32/25 | 2.5 | 1.00 | +0.93 (W0) | +0.16 (0.058) | +0.31 (0.002) | -0.13 (0.058) | +0.13 (0.115) | +0.37* / +0.29* / +0.07 / +0.06 | 1.43 | 0.76 |

* = p < 0.05 (FWER over types) in that season

figure: types_warm_eof.png

## warm days / msn: k-means on MSN a100-t30_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.27, 0.29; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W0 | 46 | +0.85 | -0.20 | +1.24 | -1.22 | -21 | 24/27/21/27 | 3.9 | 0.95 | +0.96 (W2) | +0.50 (0.002) | +0.58 (0.002) | +0.14 (0.157) | +0.44 (0.002) | +0.64* / +0.49* / +0.28* / +0.46* | 2.24 | 0.38 |
| W1 | 25 | +0.53 | +0.02 | +0.54 | +0.02 | +0 | 29/30/25/15 | 2.0 | 0.93 | +0.71 (W0) | +0.30 (0.002) | +0.43 (0.002) | +0.05 (0.843) | +0.27 (0.002) | +0.46* / +0.31* / +0.26* / +0.16 | 1.95 | 0.35 |
| W2 | 28 | +0.54 | -0.34 | +1.21 | -1.64 | -31 | 18/22/32/28 | 2.0 | 0.99 | +0.96 (W0) | +0.10 (0.425) | +0.14 (0.233) | -0.14 (0.171) | +0.09 (0.507) | +0.19 / +0.17 / +0.07 / +0.05 | 1.41 | 0.76 |

* = p < 0.05 (FWER over types) in that season

figure: types_warm_msn.png

## cold days / eof: k-means on 60 EOFs (73% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 0.99, 1.00; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 19 | -0.87 | -0.14 | -1.63 | +0.24 | +5 | 6/28/43/23 | 2.4 | 1.00 | +0.50 (C2) | -0.09 (0.278) | -0.10 (0.235) | +0.15 (0.032) | -0.09 (0.344) | -0.09 / -0.12 / -0.18 / -0.10 | 0.56 | 0.92 |
| C1 | 32 | -0.55 | +0.27 | -1.09 | +1.62 | +25 | 30/30/14/26 | 2.0 | 1.00 | +0.39 (C0) | -0.29 (0.002) | -0.35 (0.002) | +0.02 (0.982) | -0.27 (0.002) | -0.49* / -0.41* / -0.04 / -0.13 | 0.43 | 1.62 |
| C2 | 48 | -0.55 | +0.07 | -0.67 | +0.49 | +8 | 32/25/21/22 | 2.5 | 1.00 | +0.50 (C0) | -0.31 (0.002) | -0.40 (0.002) | +0.04 (0.769) | -0.29 (0.002) | -0.46* / -0.45* / -0.17 / -0.07 | 0.48 | 1.73 |

* = p < 0.05 (FWER over types) in that season

figure: types_cold_eof.png

## cold days / msn: k-means on MSN a100-t30_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.64, 0.59; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 34 | -0.74 | +0.13 | -1.30 | +1.03 | +20 | 15/21/32/32 | 2.7 | 0.98 | +0.94 (C1) | -0.18 (0.052) | -0.23 (0.002) | +0.15 (0.119) | -0.17 (0.064) | -0.41* / -0.31* / -0.15 / -0.02 | 0.56 | 1.22 |
| C1 | 34 | -0.53 | +0.18 | -1.14 | +1.42 | +23 | 24/33/22/21 | 1.8 | 0.99 | +0.94 (C0) | -0.24 (0.002) | -0.27 (0.002) | +0.07 (0.620) | -0.21 (0.014) | -0.32* / -0.28* / -0.05 / -0.23* | 0.46 | 1.48 |
| C2 | 32 | -0.57 | -0.04 | -0.52 | -0.07 | -1 | 40/29/15/16 | 2.1 | 0.99 | +0.73 (C0) | -0.32 (0.002) | -0.40 (0.002) | -0.04 (0.863) | -0.30 (0.002) | -0.45* / -0.46* / -0.20 / -0.08 | 0.41 | 1.94 |

* = p < 0.05 (FWER over types) in that season

figure: types_cold_msn.png

