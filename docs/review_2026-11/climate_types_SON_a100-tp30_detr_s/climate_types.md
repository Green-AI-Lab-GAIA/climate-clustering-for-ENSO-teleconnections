Climate types first, ENSO second (detrended anomalies, no daily-mean removal).
Warm/cold = top/bottom 20% of the daily domain-mean T anomaly (thresholds +0.33 / -0.35 sigma).
freq = days of the type / days in month; r = partial corr with the index | TNA, detrended, rotation FWER p; lift = mean freq in El Nino (index >= 0.5) or La Nina (<= -0.5) months / overall; "+lvl" also controls the month's mean T level. Types ordered by r with ONI (lag 0). Seasons: share of the type's days in DJF/MAM/JJA/SON.

## all days / eof: k-means on 60 EOFs (75% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 0.99, 0.98; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 21 | +0.67 | -0.36 | +1.34 | -1.59 | -35 | 0/0/0/100 | 4.2 | 1.00 | +0.19 (A5) | +0.49 (0.002) | +0.56 (0.002) | +0.31 (0.008) | +0.43 (0.002) | +nan / +nan / +nan / +0.49* | 1.61 | 0.67 |
| A1 | 17 | +0.08 | +0.09 | -0.07 | +0.27 | +6 | 0/0/0/100 | 2.2 | 1.00 | +0.18 (A4) | -0.06 (0.996) | -0.06 (0.996) | +0.01 (1.000) | -0.05 (0.992) | +nan / +nan / +nan / -0.06 | 0.98 | 1.03 |
| A2 | 17 | -0.11 | +0.17 | -0.52 | +1.10 | +24 | 0/0/0/100 | 1.8 | 1.00 | +0.17 (A3) | -0.10 (0.877) | -0.07 (0.990) | -0.01 (1.000) | -0.06 (0.986) | +nan / +nan / +nan / -0.10 | 0.89 | 0.96 |
| A3 | 5 | -0.69 | -0.06 | -1.25 | +0.33 | +8 | 0/0/0/100 | 2.0 | 1.00 | +0.41 (A4) | -0.13 (0.700) | -0.17 (0.511) | +0.08 (0.926) | -0.13 (0.696) | +nan / +nan / +nan / -0.13 | 0.77 | 0.96 |
| A4 | 15 | -0.43 | +0.27 | -0.80 | +1.08 | +23 | 0/0/0/100 | 2.8 | 1.00 | +0.41 (A3) | -0.21 (0.284) | -0.35 (0.002) | +0.05 (0.994) | -0.20 (0.322) | +nan / +nan / +nan / -0.21 | 0.73 | 1.12 |
| A5 | 24 | +0.03 | -0.12 | +0.38 | -0.74 | -17 | 0/0/0/100 | 2.9 | 1.00 | +0.19 (A0) | -0.24 (0.117) | -0.21 (0.247) | -0.22 (0.113) | -0.22 (0.201) | +nan / +nan / +nan / -0.24 | 0.77 | 1.23 |

* = p < 0.05 (FWER over types) in that season

figure: types_all_eof.png

## all days / msn: k-means on MSN a100-tp30_detr_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.67, 0.67; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0 | 19 | +0.72 | -0.30 | +1.36 | -1.39 | -32 | 0/0/0/100 | 4.2 | 1.00 | +0.86 (A2) | +0.44 (0.002) | +0.54 (0.002) | +0.19 (0.215) | +0.37 (0.002) | +nan / +nan / +nan / +0.44* | 1.61 | 0.70 |
| A1 | 18 | +0.10 | +0.23 | -0.22 | +1.07 | +23 | 0/0/0/100 | 2.3 | 0.98 | +0.77 (A4) | -0.04 (0.996) | +0.00 (1.000) | -0.08 (0.970) | -0.03 (0.998) | +nan / +nan / +nan / -0.04 | 0.99 | 0.99 |
| A2 | 15 | +0.10 | -0.22 | +0.28 | -0.89 | -21 | 0/0/0/100 | 1.9 | 0.93 | +0.86 (A0) | -0.06 (0.988) | -0.07 (0.980) | -0.07 (0.986) | -0.04 (0.998) | +nan / +nan / +nan / -0.06 | 1.01 | 1.03 |
| A3 | 13 | -0.58 | +0.04 | -0.78 | +0.19 | +4 | 0/0/0/100 | 3.2 | 0.99 | +0.71 (A4) | -0.09 (0.952) | -0.19 (0.300) | +0.20 (0.179) | -0.07 (0.990) | +nan / +nan / +nan / -0.09 | 0.84 | 0.95 |
| A4 | 17 | -0.35 | +0.25 | -0.91 | +1.27 | +29 | 0/0/0/100 | 2.6 | 0.99 | +0.77 (A1) | -0.19 (0.332) | -0.24 (0.085) | +0.03 (0.998) | -0.17 (0.400) | +nan / +nan / +nan / -0.19 | 0.67 | 1.03 |
| A5 | 18 | +0.03 | -0.14 | +0.46 | -0.80 | -17 | 0/0/0/100 | 2.2 | 0.94 | +0.56 (A0) | -0.23 (0.105) | -0.25 (0.046) | -0.20 (0.165) | -0.21 (0.177) | +nan / +nan / +nan / -0.23 | 0.77 | 1.33 |

* = p < 0.05 (FWER over types) in that season

figure: types_all_msn.png

## warm days / eof: k-means on 60 EOFs (73% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 0.85, 0.79; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W0 | 31 | +0.93 | -0.57 | +1.94 | -2.29 | -51 | 0/0/0/100 | 3.0 | 1.00 | +0.80 (W2) | +0.46 (0.002) | +0.48 (0.002) | +0.26 (0.060) | +0.41 (0.002) | +nan / +nan / +nan / +0.46* | 1.91 | 0.37 |
| W1 | 33 | +0.62 | -0.11 | +0.86 | -0.56 | -13 | 0/0/0/100 | 2.2 | 1.00 | +0.68 (W0) | +0.33 (0.014) | +0.33 (0.014) | +0.15 (0.390) | +0.29 (0.034) | +nan / +nan / +nan / +0.33* | 1.75 | 0.57 |
| W2 | 36 | +0.51 | -0.20 | +1.07 | -1.14 | -26 | 0/0/0/100 | 2.1 | 0.99 | +0.80 (W0) | -0.00 (1.000) | +0.19 (0.338) | -0.22 (0.137) | -0.03 (0.996) | +nan / +nan / +nan / -0.00 | 1.11 | 1.29 |

* = p < 0.05 (FWER over types) in that season

figure: types_warm_eof.png

## warm days / msn: k-means on MSN a100-tp30_detr_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.23, 0.24; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| W0 | 28 | +0.54 | -0.32 | +0.97 | -1.46 | -32 | 0/0/0/100 | 1.8 | 0.96 | +0.95 (W2) | +0.37 (0.002) | +0.40 (0.002) | +0.22 (0.107) | +0.32 (0.004) | +nan / +nan / +nan / +0.37* | 1.73 | 0.69 |
| W1 | 30 | +0.59 | +0.04 | +0.76 | -0.05 | -1 | 0/0/0/100 | 1.7 | 0.97 | +0.76 (W0) | +0.33 (0.006) | +0.45 (0.002) | +0.11 (0.640) | +0.28 (0.014) | +nan / +nan / +nan / +0.33* | 1.64 | 0.76 |
| W2 | 41 | +0.83 | -0.49 | +1.84 | -2.11 | -49 | 0/0/0/100 | 2.9 | 0.97 | +0.95 (W0) | +0.23 (0.089) | +0.30 (0.008) | -0.08 (0.787) | +0.20 (0.233) | +nan / +nan / +nan / +0.23 | 1.41 | 0.83 |

* = p < 0.05 (FWER over types) in that season

figure: types_warm_msn.png

## cold days / eof: k-means on 60 EOFs (77% var) of z~ (Tmin, Tmax, P), seeds 0-2

stability vs seeds 1, 2: ARI 0.94, 0.92; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 41 | -0.56 | +0.16 | -0.79 | +0.61 | +13 | 0/0/0/100 | 2.3 | 1.00 | +0.45 (C1) | -0.10 (0.618) | -0.24 (0.038) | +0.16 (0.217) | -0.08 (0.734) | +nan / +nan / +nan / -0.10 | 0.86 | 1.06 |
| C1 | 19 | -0.81 | -0.09 | -1.47 | +0.31 | +7 | 0/0/0/100 | 2.1 | 1.00 | +0.45 (C0) | -0.11 (0.581) | -0.14 (0.384) | +0.09 (0.793) | -0.11 (0.529) | +nan / +nan / +nan / -0.11 | 0.71 | 1.01 |
| C2 | 40 | -0.51 | +0.33 | -1.26 | +1.87 | +41 | 0/0/0/100 | 1.8 | 1.00 | +0.40 (C1) | -0.12 (0.557) | -0.11 (0.600) | +0.09 (0.767) | -0.09 (0.698) | +nan / +nan / +nan / -0.12 | 0.75 | 0.98 |

* = p < 0.05 (FWER over types) in that season

figure: types_cold_eof.png

## cold days / msn: k-means on MSN a100-tp30_detr_s0/1/2 embeddings (one per seed)

stability vs seeds 1, 2: ARI 0.65, 0.69; composite r per type (min over seeds) shown as "stab"

| type | days % | T level | P level | Tmax degC | P mm/d | P % | DJF/MAM/JJA/SON % | run (d) | stab | max pattern r with another type | r ONI (p) | r ONI lag3 (p) | r ONI +lvl (p) | r RONI (p) | r TNA-ctl seasons ONI: DJF / MAM / JJA / SON | lift EN | lift LN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 | 28 | -0.70 | -0.05 | -1.29 | +0.28 | +7 | 0/0/0/100 | 1.9 | 0.98 | +0.61 (C2) | -0.04 (0.944) | -0.08 (0.714) | +0.20 (0.044) | -0.01 (0.998) | +nan / +nan / +nan / -0.04 | 0.91 | 0.87 |
| C1 | 37 | -0.58 | +0.13 | -0.74 | +0.49 | +10 | 0/0/0/100 | 2.5 | 0.99 | +0.71 (C2) | -0.10 (0.571) | -0.19 (0.089) | +0.11 (0.457) | -0.09 (0.622) | +nan / +nan / +nan / -0.10 | 0.72 | 1.04 |
| C2 | 35 | -0.50 | +0.42 | -1.35 | +2.24 | +50 | 0/0/0/100 | 1.9 | 0.99 | +0.71 (C1) | -0.14 (0.243) | -0.16 (0.163) | +0.03 (0.966) | -0.13 (0.324) | +nan / +nan / +nan / -0.14 | 0.76 | 1.12 |

* = p < 0.05 (FWER over types) in that season

figure: types_cold_msn.png

