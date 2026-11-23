# PCA vs MSN embeddings over K (detrended Tmin, Tmax, cbrt tp; full field)

Means over 3 seeds. eta2 = share of daily field variance explained by the cluster means; split ARI = agreement of k-means fitted on even vs odd train years (PCA refitted per half; MSN embedding fixed); seed ARI = agreement across seeds (k-means seeds for PCA, independently trained models for MSN); ONI R2 = out-of-sample R2 of ONI from monthly cluster frequencies; n ONI = clusters linked to ONI | TNA (FWER); pat = median max pattern r between clusters (1 = copies).

coh = median spatial correlation of a day with its cluster map; coh>0.5 = share of days with r > 0.5.

| K | rep | eta2 | coh | coh>0.5 | pat | split ARI | seed ARI | min size % | ONI R2 (CV) | n ONI |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | MSN | 0.138 | 0.29 | 14% | 0.66 | 0.80 | 0.71 | 22.2 | 0.306 | 2.0 |
| 4 | PCA | 0.170 | 0.33 | 19% | 0.09 | 0.72 | 0.97 | 14.4 | 0.270 | 3.0 |
| 6 | MSN | 0.159 | 0.30 | 16% | 0.70 | 0.58 | 0.52 | 13.5 | 0.307 | 3.0 |
| 6 | PCA | 0.205 | 0.37 | 24% | 0.43 | 0.69 | 0.97 | 5.2 | 0.309 | 2.0 |
| 8 | MSN | 0.170 | 0.32 | 18% | 0.79 | 0.54 | 0.39 | 9.9 | 0.308 | 3.7 |
| 8 | PCA | 0.228 | 0.40 | 28% | 0.50 | 0.53 | 0.99 | 4.6 | 0.317 | 4.0 |
| 10 | MSN | 0.178 | 0.32 | 18% | 0.75 | 0.53 | 0.47 | 7.4 | 0.303 | 4.7 |
| 10 | PCA | 0.243 | 0.42 | 31% | 0.58 | 0.44 | 0.91 | 4.1 | 0.315 | 4.0 |
| 12 | MSN | 0.184 | 0.33 | 19% | 0.78 | 0.47 | 0.44 | 6.3 | 0.320 | 6.0 |
| 12 | PCA | 0.255 | 0.43 | 33% | 0.62 | 0.43 | 0.88 | 2.1 | 0.318 | 5.3 |
| 15 | MSN | 0.191 | 0.34 | 21% | 0.82 | 0.51 | 0.36 | 5.1 | 0.327 | 7.0 |
| 15 | PCA | 0.269 | 0.44 | 36% | 0.72 | 0.33 | 0.65 | 1.7 | 0.318 | 6.0 |
| 20 | MSN | 0.201 | 0.36 | 22% | 0.87 | 0.51 | 0.39 | 3.7 | 0.329 | 8.7 |
| 20 | PCA | 0.285 | 0.46 | 40% | 0.70 | 0.27 | 0.60 | 1.1 | 0.352 | 8.3 |
| 30 | MSN | 0.210 | 0.37 | 23% | 0.88 | 0.68 | 0.34 | 2.6 | 0.323 | 11.0 |
| 30 | PCA | 0.308 | 0.48 | 45% | 0.77 | 0.26 | 0.43 | 0.5 | 0.356 | 10.7 |
| 40 | MSN | 0.218 | 0.38 | 25% | 0.91 | 0.55 | 0.35 | 1.6 | 0.329 | 10.3 |
| 40 | PCA | 0.322 | 0.49 | 49% | 0.75 | 0.23 | 0.33 | 0.3 | 0.345 | 13.0 |

## Probe: ONI from monthly means of the features (no clustering; ridge, 10 contiguous folds, detrended)

| features | ONI R2 (CV) |
|---|---|
| domain-mean T level only | 0.292 |
| PCA (60 EOFs) | 0.584 |
| PCA (10 EOFs) | 0.453 |
| MSN seed 0 (384-d) | 0.376 |
| MSN seed 0, 60 PCs of the embedding | 0.372 |
| MSN seed 1 (384-d) | 0.374 |
| MSN seed 1, 60 PCs of the embedding | 0.366 |
| MSN seed 2 (384-d) | 0.360 |
| MSN seed 2, 60 PCs of the embedding | 0.356 |

## Does dimensionality matter?

| features | K | eta2 | coh | coh>0.5 | pat |
|---|---|---|---|---|---|
| PCA 20 EOFs | 10 | 0.243 | 0.42 | 31% | 0.56 |
| PCA 20 EOFs | 30 | 0.307 | 0.48 | 45% | 0.78 |
| PCA 60 EOFs | 10 | 0.243 | 0.42 | 31% | 0.61 |
| PCA 60 EOFs | 30 | 0.308 | 0.48 | 45% | 0.73 |
| PCA 200 EOFs | 10 | 0.243 | 0.42 | 31% | 0.55 |
| PCA 200 EOFs | 30 | 0.308 | 0.48 | 45% | 0.78 |
| MSN 384-d (seed 0) | 10 | 0.176 | 0.32 | 18% | 0.79 |
| MSN 384-d (seed 0) | 30 | 0.207 | 0.36 | 23% | 0.87 |
| MSN native prototypes (seed 0) | 30 | 0.207 | 0.36 | 23% | 0.87 |
