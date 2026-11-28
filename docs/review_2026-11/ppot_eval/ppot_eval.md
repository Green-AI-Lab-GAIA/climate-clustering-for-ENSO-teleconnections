# P2OT vs Sinkhorn MSN vs PCA k-means (detrended Tmin, Tmax, cbrt tp; full field)

Metrics as in mask_eval.md. used = prototypes with ≥ 1 day; n<1.5% = clusters below 1.5 % of days; size CV = sd/mean of cluster sizes. PCA K = 30 averaged over 3 k-means seeds.

| method | K | used | eta2 | coh | coh>0.5 | pat | min–max size % | n<1.5% | size CV | ONI R2 | n ONI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PCA k-means | 30 | 30 | 0.307 | 0.48 | 45% | 0.74 | 0.4–5.6 | 2 | 0.35 | 0.352 | 11.7 |
| PCA k-means | 10 | 10 | 0.243 | 0.42 | 31% | 0.61 | 4.2–13.0 | 0 | 0.26 | 0.316 | 4.0 |
| Sinkhorn + me-max (tp30_mask, ep57) native | 30 | 30 | 0.208 | 0.35 | 22% | 0.90 | 2.9–4.0 | 0 | 0.08 | 0.323 | 13.0 |
| Sinkhorn + me-max (tp30_mask, ep57) k-means | 10 | 10 | 0.177 | 0.32 | 18% | 0.77 | 9.0–11.2 | 0 | 0.07 | 0.294 | 5.0 |
| P2OT λ 3 (ep98) native | 30 | 30 | 0.213 | 0.37 | 24% | 0.90 | 2.6–4.3 | 0 | 0.12 | 0.333 | 11.0 |
| P2OT λ 3 (ep98) k-means | 10 | 10 | 0.181 | 0.33 | 18% | 0.86 | 4.2–12.4 | 0 | 0.22 | 0.305 | 6.0 |
| P2OT λ 1 (ep95) native | 30 | 30 | 0.215 | 0.37 | 24% | 0.90 | 1.6–4.2 | 0 | 0.16 | 0.334 | 10.0 |
| P2OT λ 1 (ep95) k-means | 10 | 10 | 0.177 | 0.34 | 20% | 0.75 | 8.6–12.4 | 0 | 0.11 | 0.309 | 5.0 |
| P2OT λ 0.3 (ep98) native | 30 | 22 | 0.204 | 0.35 | 21% | 0.79 | 2.3–7.0 | 0 | 0.68 | 0.345 | 8.0 |
| P2OT λ 0.3 (ep98) k-means | 10 | 10 | 0.181 | 0.33 | 18% | 0.74 | 5.1–11.8 | 0 | 0.19 | 0.338 | 5.0 |

## Rare EOF types (EOF k-means K = 30, seed 0, 8 smallest): best-matching native cluster per run (pattern r, size %)

| EOF size % | Tmin °C | Tmax °C | P % | Sinkhorn + me-max (tp30_mask, ep57) | P2OT λ 3 (ep98) | P2OT λ 1 (ep95) | P2OT λ 0.3 (ep98) |
|---|---|---|---|---|---|---|---|
| 0.3 | -2.65 | -2.68 | -12 | 0.84 (3.7) | 0.85 (3.3) | 0.85 (3.4) | 0.86 (4.2) |
| 0.9 | +0.26 | -0.57 | +32 | 0.71 (3.5) | 0.67 (3.3) | 0.71 (1.6) | 0.64 (5.0) |
| 1.5 | -1.23 | -1.05 | -1 | 0.87 (3.7) | 0.89 (3.3) | 0.89 (3.4) | 0.86 (4.2) |
| 1.5 | -1.22 | -1.80 | +8 | 0.92 (3.2) | 0.94 (3.0) | 0.95 (3.2) | 0.89 (5.2) |
| 2.0 | -0.37 | -0.99 | +13 | 0.71 (3.2) | 0.84 (3.0) | 0.83 (3.2) | 0.85 (5.2) |
| 2.0 | -0.54 | -1.07 | +23 | 0.78 (3.2) | 0.80 (3.2) | 0.79 (3.3) | 0.79 (4.7) |
| 2.6 | -0.26 | -0.56 | +10 | 0.51 (3.5) | 0.70 (2.9) | 0.86 (2.8) | 0.51 (5.0) |
| 2.8 | -0.43 | -0.19 | -10 | 0.79 (3.2) | 0.78 (3.3) | 0.76 (3.2) | 0.74 (5.2) |

median best-match r over all 30 EOF types: Sinkhorn + me-max (tp30_mask, ep57) 0.76, P2OT λ 3 (ep98) 0.79, P2OT λ 1 (ep95) 0.80, P2OT λ 0.3 (ep98) 0.77
median best-match r, 8 smallest EOF types: Sinkhorn + me-max (tp30_mask, ep57) 0.78, P2OT λ 3 (ep98) 0.82, P2OT λ 1 (ep95) 0.84, P2OT λ 0.3 (ep98) 0.82
