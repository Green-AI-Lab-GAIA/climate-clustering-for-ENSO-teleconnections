Climate types first, ENSO second (detrended anomalies, no daily-mean removal).
Warm/cold = top/bottom 20% of the daily domain-mean T anomaly (thresholds +0.33 / -0.35 sigma).
freq = days of the type / days in month; r = partial corr with the index | TNA, detrended, rotation FWER p; lift = mean freq in El Nino (index >= 0.5) or La Nina (<= -0.5) months / overall; "+lvl" also controls the month's mean T level. Types ordered by r with ONI (lag 0). Seasons: share of the type's days in DJF/MAM/JJA/SON.

## all days / eof: k-means on 60 EOFs (75% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 0.97, 0.99; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 21 | +0.67 | -0.36 | +1.34 | -1.59 | -36 | 0/0/0/100 | 4.2 | 1.00 | +0.19 (A5) | +0.48 (0.002) | +0.56 (0.002) | +0.31 (0.010) | +0.43 (0.002) | +nan / +nan / +nan / +0.48* | 1.61 | 0.67 |
| A1 | 17 | +0.08 | +0.10 | -0.07 | +0.28 | +6 | 0/0/0/100 | 2.2 | 1.00 | +0.19 (A4) | -0.05 (0.998) | -0.06 (0.996) | +0.01 (1.000) | -0.05 (0.996) | +nan / +nan / +nan / -0.05 | 0.99 | 1.02 |
| A2 | 17 | -0.10 | +0.16 | -0.51 | +1.08 | +23 | 0/0/0/100 | 1.8 | 1.00 | +0.16 (A3) | -0.09 (0.893) | -0.05 (0.998) | -0.02 (1.000) | -0.05 (0.992) | +nan / +nan / +nan / -0.09 | 0.91 | 0.97 |
| A3 | 5 | -0.69 | -0.06 | -1.24 | +0.30 | +7 | 0/0/0/100 | 2.1 | 1.00 | +0.42 (A4) | -0.14 (0.622) | -0.18 (0.481) | +0.07 (0.966) | -0.15 (0.638) | +nan / +nan / +nan / -0.14 | 0.74 | 0.98 |
| A4 | 15 | -0.43 | +0.28 | -0.82 | +1.13 | +24 | 0/0/0/100 | 2.9 | 1.00 | +0.42 (A3) | -0.20 (0.362) | -0.34 (0.002) | +0.06 (0.986) | -0.19 (0.435) | +nan / +nan / +nan / -0.20 | 0.74 | 1.11 |
| A5 | 24 | +0.03 | -0.12 | +0.38 | -0.74 | -17 | 0/0/0/100 | 2.9 | 1.00 | +0.19 (A0) | -0.25 (0.125) | -0.22 (0.235) | -0.22 (0.121) | -0.22 (0.201) | +nan / +nan / +nan / -0.25 | 0.76 | 1.24 |

* = p < 0.05 (FWER over types) in that season

figure: types_all_eof.png

## all days / msn: k-means on MSN a100-t30_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.59, 0.65; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 19 | +0.69 | -0.29 | +1.30 | -1.36 | -31 | 0/0/0/100 | 4.1 | 0.96 | +0.78 (A5) | +0.39 (0.002) | +0.45 (0.002) | +0.11 (0.755) | +0.34 (0.006) | +nan / +nan / +nan / +0.39* | 1.55 | 0.74 |
| A1 | 17 | +0.17 | +0.12 | -0.02 | +0.58 | +12 | 0/0/0/100 | 2.4 | 0.84 | +0.59 (A4) | +0.01 (1.000) | +0.10 (0.859) | -0.03 (0.998) | +0.01 (1.000) | +nan / +nan / +nan / +0.01 | 1.15 | 1.13 |
| A2 | 15 | -0.55 | +0.13 | -0.97 | +0.69 | +16 | 0/0/0/100 | 3.5 | 0.99 | +0.76 (A4) | -0.05 (0.990) | -0.12 (0.777) | +0.17 (0.310) | -0.08 (0.968) | +nan / +nan / +nan / -0.05 | 0.74 | 0.80 |
| A3 | 15 | -0.16 | -0.09 | +0.02 | -0.57 | -12 | 0/0/0/100 | 2.0 | 0.83 | +0.57 (A5) | -0.11 (0.803) | -0.21 (0.217) | +0.05 (0.990) | -0.08 (0.944) | +nan / +nan / +nan / -0.11 | 0.88 | 1.02 |
| A4 | 17 | -0.21 | +0.26 | -0.70 | +1.36 | +30 | 0/0/0/100 | 2.1 | 0.96 | +0.76 (A2) | -0.18 (0.334) | -0.21 (0.175) | -0.01 (1.000) | -0.15 (0.549) | +nan / +nan / +nan / -0.18 | 0.75 | 0.98 |
| A5 | 16 | +0.16 | -0.24 | +0.65 | -1.17 | -26 | 0/0/0/100 | 2.2 | 0.98 | +0.78 (A0) | -0.22 (0.127) | -0.22 (0.127) | -0.24 (0.034) | -0.18 (0.298) | +nan / +nan / +nan / -0.22 | 0.85 | 1.37 |

* = p < 0.05 (FWER over types) in that season

figure: types_all_msn.png

## warm days / eof: k-means on 60 EOFs (73% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 0.94, 0.93; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W0 | 32 | +0.92 | -0.57 | +1.94 | -2.34 | -52 | 0/0/0/100 | 3.0 | 1.00 | +0.79 (W2) | +0.45 (0.002) | +0.48 (0.002) | +0.26 (0.072) | +0.41 (0.002) | +nan / +nan / +nan / +0.45* | 1.88 | 0.40 |
| W1 | 33 | +0.62 | -0.11 | +0.87 | -0.58 | -13 | 0/0/0/100 | 2.2 | 1.00 | +0.68 (W0) | +0.34 (0.012) | +0.34 (0.014) | +0.16 (0.338) | +0.30 (0.038) | +nan / +nan / +nan / +0.34* | 1.75 | 0.55 |
| W2 | 35 | +0.50 | -0.18 | +1.02 | -1.05 | -24 | 0/0/0/100 | 2.1 | 1.00 | +0.79 (W0) | -0.01 (1.000) | +0.18 (0.386) | -0.22 (0.119) | -0.05 (0.988) | +nan / +nan / +nan / -0.01 | 1.12 | 1.32 |

* = p < 0.05 (FWER over types) in that season

figure: types_warm_eof.png

## warm days / msn: k-means on MSN a100-t30_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.25, 0.53; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W0 | 47 | +0.88 | -0.36 | +1.61 | -1.69 | -38 | 0/0/0/100 | 3.7 | 0.87 | +0.94 (W2) | +0.43 (0.002) | +0.46 (0.002) | +0.20 (0.137) | +0.37 (0.002) | +nan / +nan / +nan / +0.43* | 1.83 | 0.49 |
| W1 | 24 | +0.48 | +0.02 | +0.53 | +0.09 | +2 | 0/0/0/100 | 1.6 | 0.87 | +0.66 (W0) | +0.22 (0.163) | +0.34 (0.006) | +0.09 (0.732) | +0.18 (0.318) | +nan / +nan / +nan / +0.22 | 1.65 | 0.85 |
| W2 | 29 | +0.51 | -0.41 | +1.32 | -1.85 | -41 | 0/0/0/100 | 1.9 | 0.98 | +0.94 (W0) | +0.01 (0.990) | +0.09 (0.805) | -0.18 (0.256) | +0.00 (1.000) | +nan / +nan / +nan / +0.01 | 1.10 | 1.14 |

* = p < 0.05 (FWER over types) in that season

figure: types_warm_msn.png

## cold days / eof: k-means on 60 EOFs (77% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 0.97, 0.98; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 42 | -0.56 | +0.17 | -0.81 | +0.68 | +14 | 0/0/0/100 | 2.3 | 1.00 | +0.45 (C1) | -0.10 (0.652) | -0.24 (0.044) | +0.17 (0.209) | -0.08 (0.761) | +nan / +nan / +nan / -0.10 | 0.88 | 1.07 |
| C1 | 19 | -0.81 | -0.09 | -1.47 | +0.30 | +7 | 0/0/0/100 | 2.1 | 1.00 | +0.45 (C0) | -0.11 (0.579) | -0.16 (0.284) | +0.09 (0.740) | -0.11 (0.539) | +nan / +nan / +nan / -0.11 | 0.72 | 0.99 |
| C2 | 39 | -0.51 | +0.32 | -1.26 | +1.83 | +40 | 0/0/0/100 | 1.7 | 1.00 | +0.40 (C1) | -0.12 (0.535) | -0.11 (0.630) | +0.08 (0.805) | -0.09 (0.672) | +nan / +nan / +nan / -0.12 | 0.72 | 0.97 |

* = p < 0.05 (FWER over types) in that season

figure: types_cold_eof.png

## cold days / msn: k-means on MSN a100-t30_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.33, 0.63; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 34 | -0.58 | +0.30 | -1.34 | +1.53 | +36 | 0/0/0/100 | 2.0 | 0.93 | +0.96 (C2) | -0.03 (0.990) | -0.03 (0.952) | +0.15 (0.264) | -0.02 (0.986) | +nan / +nan / +nan / -0.03 | 0.92 | 0.85 |
| C1 | 39 | -0.65 | -0.02 | -0.80 | -0.02 | -0 | 0/0/0/100 | 2.3 | 0.97 | +0.70 (C0) | -0.08 (0.746) | -0.19 (0.117) | +0.17 (0.179) | -0.06 (0.859) | +nan / +nan / +nan / -0.08 | 0.79 | 0.90 |
| C2 | 27 | -0.50 | +0.32 | -1.25 | +1.98 | +41 | 0/0/0/100 | 1.6 | 0.99 | +0.96 (C0) | -0.23 (0.032) | -0.27 (0.012) | -0.05 (0.928) | -0.20 (0.093) | +nan / +nan / +nan / -0.23* | 0.61 | 1.40 |

* = p < 0.05 (FWER over types) in that season

figure: types_cold_msn.png

