Climate types first, ENSO second (detrended anomalies, no daily-mean removal).
Warm/cold = top/bottom 20% of the daily domain-mean T anomaly (thresholds +0.39 / -0.38 sigma).
freq = days of the type / days in month; r = partial corr with the index | TNA, detrended, rotation FWER p; lift = mean freq in El Nino (index >= 0.5) or La Nina (<= -0.5) months / overall; "+lvl" also controls the month's mean T level. Types ordered by r with ONI (lag 0). Seasons: share of the type's days in DJF/MAM/JJA/SON.

## all days / eof: k-means on 60 EOFs (74% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 0.91, 0.93; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 11 | +0.94 | -0.17 | +1.15 | -1.08 | -14 | 0/100/0/0 | 5.7 | 1.00 | +0.63 (A2) | +0.48 (0.002) | +0.64 (0.002) | +0.01 (1.000) | +0.43 (0.002) | +nan / +0.48* / +nan / +nan | 2.93 | 0.09 |
| A1 | 18 | +0.26 | +0.01 | +0.27 | -0.34 | -4 | 0/100/0/0 | 2.9 | 1.00 | +0.47 (A4) | +0.28 (0.048) | +0.29 (0.026) | +0.08 (0.934) | +0.22 (0.223) | +nan / +0.28* / +nan / +nan | 1.23 | 0.47 |
| A2 | 22 | +0.23 | -0.10 | +0.47 | -0.43 | -6 | 0/100/0/0 | 3.3 | 1.00 | +0.63 (A0) | +0.09 (0.934) | +0.03 (1.000) | +0.01 (1.000) | +0.08 (0.966) | +nan / +0.09 / +nan / +nan | 0.93 | 0.60 |
| A3 | 7 | -0.65 | -0.18 | -1.12 | +0.14 | +2 | 0/100/0/0 | 2.8 | 1.00 | +0.21 (A5) | -0.12 (0.789) | -0.07 (0.978) | +0.10 (0.867) | -0.11 (0.889) | +nan / -0.12 / +nan / +nan | 0.74 | 0.97 |
| A4 | 19 | -0.31 | +0.17 | -0.52 | +0.76 | +10 | 0/100/0/0 | 2.9 | 1.00 | +0.47 (A1) | -0.34 (0.010) | -0.44 (0.002) | -0.02 (1.000) | -0.29 (0.036) | +nan / -0.34* / +nan / +nan | 0.47 | 1.40 |
| A5 | 23 | -0.31 | +0.05 | -0.33 | +0.43 | +6 | 0/100/0/0 | 3.3 | 1.00 | +0.36 (A2) | -0.51 (0.002) | -0.55 (0.002) | -0.15 (0.553) | -0.45 (0.002) | +nan / -0.51* / +nan / +nan | 0.47 | 1.90 |

* = p < 0.05 (FWER over types) in that season

figure: types_all_eof.png

## all days / msn: k-means on MSN a100-t30_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.68, 0.71; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 18 | +0.73 | -0.10 | +0.87 | -0.73 | -10 | 0/100/0/0 | 6.1 | 0.99 | +0.73 (A2) | +0.51 (0.002) | +0.62 (0.002) | +0.12 (0.809) | +0.43 (0.002) | +nan / +0.51* / +nan / +nan | 2.42 | 0.12 |
| A1 | 15 | +0.19 | +0.13 | -0.01 | +0.70 | +9 | 0/100/0/0 | 2.4 | 0.73 | +0.72 (A3) | +0.28 (0.058) | +0.31 (0.020) | +0.19 (0.366) | +0.25 (0.143) | +nan / +0.28 / +nan / +nan | 1.23 | 0.41 |
| A2 | 18 | +0.17 | -0.14 | +0.48 | -0.89 | -12 | 0/100/0/0 | 2.3 | 0.98 | +0.78 (A5) | -0.00 (1.000) | -0.03 (1.000) | -0.02 (1.000) | +0.02 (1.000) | +nan / -0.00 / +nan / +nan | 0.78 | 0.89 |
| A3 | 18 | -0.24 | +0.16 | -0.57 | +1.07 | +14 | 0/100/0/0 | 2.3 | 0.99 | +0.80 (A4) | -0.15 (0.608) | -0.21 (0.276) | +0.06 (0.998) | -0.11 (0.869) | +nan / -0.15 / +nan / +nan | 0.65 | 1.07 |
| A4 | 14 | -0.60 | +0.02 | -0.79 | +0.45 | +6 | 0/100/0/0 | 3.6 | 1.00 | +0.80 (A3) | -0.44 (0.002) | -0.48 (0.002) | -0.13 (0.730) | -0.40 (0.002) | +nan / -0.44* / +nan / +nan | 0.52 | 1.90 |
| A5 | 17 | -0.18 | -0.09 | +0.04 | -0.62 | -8 | 0/100/0/0 | 2.4 | 0.98 | +0.78 (A2) | -0.44 (0.002) | -0.52 (0.002) | -0.24 (0.141) | -0.38 (0.002) | +nan / -0.44* / +nan / +nan | 0.42 | 1.67 |

* = p < 0.05 (FWER over types) in that season

figure: types_all_msn.png

## warm days / eof: k-means on 60 EOFs (73% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 0.94, 0.97; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W0 | 30 | +1.04 | -0.13 | +1.16 | -0.87 | -11 | 0/100/0/0 | 4.6 | 1.00 | +0.83 (W2) | +0.39 (0.002) | +0.54 (0.002) | -0.04 (0.986) | +0.34 (0.002) | +nan / +0.39* / +nan / +nan | 3.10 | 0.13 |
| W1 | 33 | +0.62 | -0.04 | +0.77 | -0.77 | -10 | 0/100/0/0 | 2.7 | 1.00 | +0.32 (W0) | +0.34 (0.002) | +0.39 (0.002) | +0.03 (0.996) | +0.28 (0.018) | +nan / +0.34* / +nan / +nan | 1.98 | 0.17 |
| W2 | 37 | +0.59 | -0.21 | +0.98 | -1.11 | -15 | 0/100/0/0 | 2.4 | 1.00 | +0.83 (W0) | +0.30 (0.002) | +0.35 (0.002) | +0.06 (0.924) | +0.25 (0.040) | +nan / +0.30* / +nan / +nan | 1.87 | 0.20 |

* = p < 0.05 (FWER over types) in that season

figure: types_warm_eof.png

## warm days / msn: k-means on MSN a100-t30_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.63, 0.72; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W0 | 52 | +0.89 | -0.12 | +1.06 | -0.92 | -12 | 0/100/0/0 | 5.1 | 0.99 | +0.92 (W2) | +0.46 (0.002) | +0.60 (0.002) | +0.06 (0.918) | +0.39 (0.002) | +nan / +0.46* / +nan / +nan | 2.70 | 0.03 |
| W1 | 26 | +0.55 | +0.05 | +0.56 | +0.08 | +1 | 0/100/0/0 | 2.2 | 0.96 | +0.66 (W0) | +0.21 (0.181) | +0.32 (0.026) | -0.03 (0.994) | +0.18 (0.286) | +nan / +0.21 / +nan / +nan | 1.68 | 0.33 |
| W2 | 22 | +0.59 | -0.35 | +1.18 | -2.08 | -29 | 0/100/0/0 | 2.0 | 1.00 | +0.92 (W0) | +0.18 (0.316) | +0.13 (0.503) | -0.02 (0.998) | +0.16 (0.374) | +nan / +0.18 / +nan / +nan | 2.00 | 0.32 |

* = p < 0.05 (FWER over types) in that season

figure: types_warm_msn.png

## cold days / eof: k-means on 60 EOFs (75% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 1.00, 1.00; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 21 | -0.81 | -0.22 | -1.40 | +0.16 | +2 | 0/100/0/0 | 2.5 | 1.00 | +0.41 (C2) | -0.09 (0.678) | -0.06 (0.845) | +0.12 (0.495) | -0.08 (0.779) | +nan / -0.09 / +nan / +nan | 0.69 | 0.96 |
| C1 | 36 | -0.57 | +0.15 | -0.89 | +0.99 | +13 | 0/100/0/0 | 2.3 | 1.00 | +0.25 (C0) | -0.40 (0.002) | -0.43 (0.002) | -0.04 (0.960) | -0.35 (0.002) | +nan / -0.40* / +nan / +nan | 0.27 | 1.92 |
| C2 | 43 | -0.58 | +0.08 | -0.71 | +0.79 | +10 | 0/100/0/0 | 2.5 | 1.00 | +0.41 (C0) | -0.44 (0.002) | -0.48 (0.002) | -0.10 (0.581) | -0.39 (0.002) | +nan / -0.44* / +nan / +nan | 0.35 | 2.45 |

* = p < 0.05 (FWER over types) in that season

figure: types_cold_eof.png

## cold days / msn: k-means on MSN a100-t30_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.64, 0.60; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 40 | -0.56 | +0.08 | -1.03 | +1.16 | +15 | 0/100/0/0 | 2.1 | 0.99 | +0.94 (C1) | -0.22 (0.091) | -0.21 (0.117) | +0.08 (0.847) | -0.19 (0.223) | +nan / -0.22 / +nan / +nan | 0.39 | 1.52 |
| C1 | 33 | -0.74 | +0.08 | -1.12 | +1.00 | +14 | 0/100/0/0 | 2.7 | 0.98 | +0.94 (C0) | -0.36 (0.002) | -0.42 (0.002) | -0.03 (0.994) | -0.33 (0.004) | +nan / -0.36* / +nan / +nan | 0.49 | 2.00 |
| C2 | 27 | -0.58 | -0.06 | -0.49 | -0.24 | -3 | 0/100/0/0 | 1.8 | 0.98 | +0.67 (C1) | -0.45 (0.002) | -0.46 (0.002) | -0.14 (0.483) | -0.39 (0.002) | +nan / -0.45* / +nan / +nan | 0.26 | 2.52 |

* = p < 0.05 (FWER over types) in that season

figure: types_cold_msn.png

