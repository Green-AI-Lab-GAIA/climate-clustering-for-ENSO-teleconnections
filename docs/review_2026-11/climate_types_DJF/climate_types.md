Climate types first, ENSO second (detrended anomalies, no daily-mean removal).
Warm/cold = top/bottom 20% of the daily domain-mean T anomaly (thresholds +0.35 / -0.37 sigma).
freq = days of the type / days in month; r = partial corr with the index | TNA, detrended, rotation FWER p; lift = mean freq in El Nino (index >= 0.5) or La Nina (<= -0.5) months / overall; "+lvl" also controls the month's mean T level. Types ordered by r with ONI (lag 0). Seasons: share of the type's days in DJF/MAM/JJA/SON.

## all days / eof: k-means on 60 EOFs (73% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 0.99, 0.97; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 12 | +0.76 | -0.23 | +1.10 | -1.78 | -22 | 100/0/0/0 | 3.9 | 1.00 | +0.52 (A3) | +0.64 (0.002) | +0.67 (0.002) | +0.04 (0.988) | +0.60 (0.002) | +0.64* / +nan / +nan / +nan | 2.26 | 0.25 |
| A1 | 11 | +0.51 | -0.19 | +0.56 | -0.44 | -5 | 100/0/0/0 | 3.7 | 1.00 | +0.62 (A4) | +0.49 (0.002) | +0.58 (0.002) | +0.08 (0.905) | +0.48 (0.002) | +0.49* / +nan / +nan / +nan | 2.15 | 0.19 |
| A2 | 24 | +0.05 | +0.08 | -0.09 | +0.31 | +4 | 100/0/0/0 | 2.8 | 1.00 | +0.06 (A4) | -0.05 (0.988) | -0.07 (0.944) | -0.06 (0.950) | -0.02 (1.000) | -0.05 / +nan / +nan / +nan | 0.91 | 0.88 |
| A3 | 19 | -0.04 | -0.06 | +0.22 | -0.82 | -10 | 100/0/0/0 | 3.7 | 1.00 | +0.52 (A0) | -0.24 (0.125) | -0.29 (0.036) | -0.04 (0.990) | -0.24 (0.127) | -0.24 / +nan / +nan / +nan | 0.70 | 1.17 |
| A4 | 15 | -0.27 | +0.04 | -0.49 | +1.03 | +13 | 100/0/0/0 | 3.0 | 1.00 | +0.62 (A1) | -0.33 (0.008) | -0.31 (0.012) | -0.07 (0.936) | -0.30 (0.042) | -0.33* / +nan / +nan / +nan | 0.52 | 1.31 |
| A5 | 19 | -0.52 | +0.17 | -0.68 | +0.95 | +12 | 100/0/0/0 | 3.7 | 1.00 | +0.19 (A3) | -0.51 (0.002) | -0.53 (0.002) | +0.08 (0.881) | -0.51 (0.002) | -0.51* / +nan / +nan / +nan | 0.30 | 1.68 |

* = p < 0.05 (FWER over types) in that season

figure: types_all_eof.png

## all days / msn: k-means on MSN a100-t30_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.79, 0.81; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 18 | +0.71 | -0.16 | +0.84 | -1.11 | -14 | 100/0/0/0 | 5.5 | 1.00 | +0.69 (A2) | +0.66 (0.002) | +0.72 (0.002) | +0.10 (0.905) | +0.61 (0.002) | +0.66* / +nan / +nan / +nan | 2.34 | 0.27 |
| A1 | 17 | +0.09 | +0.11 | -0.20 | +1.05 | +13 | 100/0/0/0 | 2.6 | 0.98 | +0.76 (A4) | +0.18 (0.567) | +0.19 (0.465) | +0.04 (0.996) | +0.17 (0.608) | +0.18 / +nan / +nan / +nan | 1.27 | 0.68 |
| A2 | 15 | +0.19 | -0.13 | +0.42 | -1.12 | -14 | 100/0/0/0 | 2.2 | 0.96 | +0.74 (A3) | +0.04 (0.998) | -0.00 (1.000) | +0.06 (0.982) | +0.06 (0.988) | +0.04 / +nan / +nan / +nan | 1.11 | 0.79 |
| A3 | 15 | -0.11 | -0.08 | +0.15 | -0.88 | -11 | 100/0/0/0 | 2.1 | 1.00 | +0.74 (A2) | -0.32 (0.014) | -0.38 (0.002) | -0.04 (1.000) | -0.28 (0.066) | -0.32* / +nan / +nan / +nan | 0.62 | 1.24 |
| A4 | 18 | -0.27 | +0.14 | -0.54 | +1.23 | +15 | 100/0/0/0 | 2.3 | 1.00 | +0.76 (A1) | -0.37 (0.002) | -0.38 (0.002) | -0.07 (0.974) | -0.32 (0.014) | -0.37* / +nan / +nan / +nan | 0.48 | 1.27 |
| A5 | 18 | -0.49 | +0.08 | -0.49 | +0.45 | +6 | 100/0/0/0 | 4.8 | 1.00 | +0.52 (A4) | -0.45 (0.002) | -0.46 (0.002) | -0.09 (0.944) | -0.45 (0.002) | -0.45* / +nan / +nan / +nan | 0.21 | 1.70 |

* = p < 0.05 (FWER over types) in that season

figure: types_all_msn.png

## warm days / eof: k-means on 60 EOFs (74% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 0.99, 0.99; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W0 | 23 | +1.07 | -0.32 | +1.44 | -2.19 | -27 | 100/0/0/0 | 3.8 | 1.00 | +0.86 (W2) | +0.56 (0.002) | +0.61 (0.002) | +0.01 (1.000) | +0.53 (0.002) | +0.56* / +nan / +nan / +nan | 2.86 | 0.07 |
| W1 | 33 | +0.64 | -0.19 | +0.70 | -0.53 | -6 | 100/0/0/0 | 3.2 | 1.00 | +0.42 (W0) | +0.49 (0.002) | +0.58 (0.002) | +0.05 (0.905) | +0.46 (0.002) | +0.49* / +nan / +nan / +nan | 2.38 | 0.13 |
| W2 | 44 | +0.55 | -0.16 | +0.85 | -1.42 | -18 | 100/0/0/0 | 2.8 | 1.00 | +0.86 (W0) | +0.43 (0.002) | +0.44 (0.002) | +0.03 (0.984) | +0.40 (0.002) | +0.43* / +nan / +nan / +nan | 1.85 | 0.37 |

* = p < 0.05 (FWER over types) in that season

figure: types_warm_eof.png

## warm days / msn: k-means on MSN a100-t30_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.43, 0.54; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W0 | 49 | +0.89 | -0.22 | +1.11 | -1.57 | -19 | 100/0/0/0 | 4.9 | 0.97 | +0.92 (W2) | +0.62 (0.002) | +0.66 (0.002) | +0.09 (0.751) | +0.57 (0.002) | +0.62* / +nan / +nan / +nan | 2.68 | 0.18 |
| W1 | 27 | +0.54 | -0.09 | +0.56 | -0.24 | -3 | 100/0/0/0 | 1.9 | 0.93 | +0.74 (W0) | +0.43 (0.002) | +0.52 (0.002) | -0.01 (1.000) | +0.41 (0.002) | +0.43* / +nan / +nan / +nan | 2.09 | 0.13 |
| W2 | 24 | +0.51 | -0.32 | +1.02 | -1.97 | -25 | 100/0/0/0 | 2.1 | 0.99 | +0.92 (W0) | +0.20 (0.211) | +0.20 (0.209) | -0.04 (0.952) | +0.22 (0.147) | +0.20 / +nan / +nan / +nan | 1.59 | 0.41 |

* = p < 0.05 (FWER over types) in that season

figure: types_warm_msn.png

## cold days / eof: k-means on 60 EOFs (71% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 0.96, 0.97; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 29 | -0.59 | +0.21 | -0.94 | +1.44 | +18 | 100/0/0/0 | 2.1 | 1.00 | +0.41 (C2) | -0.37 (0.002) | -0.42 (0.002) | +0.09 (0.767) | -0.36 (0.002) | -0.37* / +nan / +nan / +nan | 0.29 | 1.78 |
| C1 | 31 | -0.57 | +0.13 | -0.91 | +1.64 | +20 | 100/0/0/0 | 2.2 | 1.00 | +0.36 (C0) | -0.40 (0.002) | -0.42 (0.002) | +0.02 (0.996) | -0.39 (0.002) | -0.40* / +nan / +nan / +nan | 0.23 | 1.80 |
| C2 | 40 | -0.56 | +0.13 | -0.58 | +0.55 | +7 | 100/0/0/0 | 2.6 | 1.00 | +0.41 (C0) | -0.41 (0.002) | -0.38 (0.002) | +0.01 (1.000) | -0.41 (0.002) | -0.41* / +nan / +nan / +nan | 0.22 | 1.75 |

* = p < 0.05 (FWER over types) in that season

figure: types_cold_eof.png

## cold days / msn: k-means on MSN a100-t30_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.34, 0.76; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 24 | -0.52 | +0.25 | -1.05 | +2.25 | +28 | 100/0/0/0 | 1.8 | 0.98 | +0.88 (C1) | -0.29 (0.046) | -0.31 (0.028) | +0.09 (0.763) | -0.28 (0.064) | -0.29* / +nan / +nan / +nan | 0.37 | 1.68 |
| C1 | 37 | -0.55 | +0.14 | -0.78 | +1.07 | +13 | 100/0/0/0 | 2.1 | 0.98 | +0.88 (C0) | -0.35 (0.008) | -0.36 (0.006) | +0.18 (0.336) | -0.34 (0.008) | -0.35* / +nan / +nan / +nan | 0.34 | 1.65 |
| C2 | 40 | -0.63 | +0.11 | -0.64 | +0.57 | +7 | 100/0/0/0 | 3.4 | 0.98 | +0.84 (C1) | -0.41 (0.002) | -0.41 (0.002) | -0.09 (0.775) | -0.42 (0.002) | -0.41* / +nan / +nan / +nan | 0.08 | 1.95 |

* = p < 0.05 (FWER over types) in that season

figure: types_cold_msn.png

