# e-means vs Lloyd k-means (means over 3 seeds)

rel inertia = train k-means objective / best of the 3 Lloyd runs (1 = as good as Lloyd; < 1 = better); eta2, coh (median day-to-map r), pat (median max pattern r between maps) as in k_selection; ARI Lloyd = daily-label agreement with the Lloyd run of the same seed.

| features | K | method | rel inertia | eta2 | coh | pat | seed ARI | ARI vs Lloyd | min size % |
|---|---|---|---|---|---|---|---|---|---|
| PCA 60 EOFs | 10 | Lloyd (sklearn, n_init 10) | 1.0000 | 0.243 | 0.416 | 0.57 | 0.90 | 1.00 | 4.1 |
| PCA 60 EOFs | 10 | e-means a0=0.03 B=256 T=3 | 1.0132 | 0.238 | 0.413 | 0.57 | 0.35 | 0.38 | 4.6 |
| PCA 60 EOFs | 10 | e-means a0=0.03 B=256 T=10 | 1.0062 | 0.240 | 0.416 | 0.59 | 0.43 | 0.47 | 4.5 |
| PCA 60 EOFs | 10 | e-means a0=0.03 B=1024 T=3 | 1.0197 | 0.235 | 0.410 | 0.54 | 0.34 | 0.33 | 4.6 |
| PCA 60 EOFs | 10 | e-means a0=0.03 B=1024 T=10 | 1.0139 | 0.237 | 0.414 | 0.54 | 0.38 | 0.36 | 4.6 |
| PCA 60 EOFs | 10 | e-means a0=0.1 B=256 T=3 | 1.0068 | 0.240 | 0.416 | 0.60 | 0.41 | 0.45 | 4.4 |
| PCA 60 EOFs | 10 | e-means a0=0.1 B=256 T=10 | 1.0012 | 0.243 | 0.416 | 0.56 | 0.51 | 0.66 | 3.9 |
| PCA 60 EOFs | 10 | e-means a0=0.1 B=1024 T=3 | 1.0170 | 0.236 | 0.412 | 0.53 | 0.36 | 0.34 | 4.6 |
| PCA 60 EOFs | 10 | e-means a0=0.1 B=1024 T=10 | 1.0069 | 0.240 | 0.418 | 0.58 | 0.45 | 0.43 | 4.5 |
| PCA 60 EOFs | 10 | e-means a0=0.3 B=256 T=3 | 1.0028 | 0.242 | 0.418 | 0.56 | 0.53 | 0.59 | 4.1 |
| PCA 60 EOFs | 10 | e-means a0=0.3 B=256 T=10 | 1.0011 | 0.242 | 0.418 | 0.51 | 0.56 | 0.62 | 3.7 |
| PCA 60 EOFs | 10 | e-means a0=0.3 B=1024 T=3 | 1.0095 | 0.239 | 0.417 | 0.55 | 0.41 | 0.39 | 4.6 |
| PCA 60 EOFs | 10 | e-means a0=0.3 B=1024 T=10 | 1.0023 | 0.242 | 0.417 | 0.61 | 0.55 | 0.59 | 4.1 |
| PCA 60 EOFs | 30 | Lloyd (sklearn, n_init 10) | 1.0001 | 0.308 | 0.480 | 0.75 | 0.39 | 1.00 | 0.5 |
| PCA 60 EOFs | 30 | e-means a0=0.03 B=256 T=3 | 1.0104 | 0.304 | 0.478 | 0.74 | 0.25 | 0.31 | 0.7 |
| PCA 60 EOFs | 30 | e-means a0=0.03 B=256 T=10 | 1.0059 | 0.305 | 0.479 | 0.74 | 0.28 | 0.34 | 0.6 |
| PCA 60 EOFs | 30 | e-means a0=0.03 B=1024 T=3 | 1.0158 | 0.301 | 0.477 | 0.73 | 0.24 | 0.28 | 0.9 |
| PCA 60 EOFs | 30 | e-means a0=0.03 B=1024 T=10 | 1.0114 | 0.303 | 0.479 | 0.73 | 0.25 | 0.31 | 0.9 |
| PCA 60 EOFs | 30 | e-means a0=0.1 B=256 T=3 | 1.0050 | 0.306 | 0.480 | 0.73 | 0.30 | 0.32 | 0.7 |
| PCA 60 EOFs | 30 | e-means a0=0.1 B=256 T=10 | 1.0018 | 0.307 | 0.480 | 0.77 | 0.33 | 0.34 | 0.4 |
| PCA 60 EOFs | 30 | e-means a0=0.1 B=1024 T=3 | 1.0136 | 0.302 | 0.479 | 0.72 | 0.24 | 0.29 | 0.9 |
| PCA 60 EOFs | 30 | e-means a0=0.1 B=1024 T=10 | 1.0075 | 0.305 | 0.480 | 0.73 | 0.27 | 0.33 | 0.8 |
| PCA 60 EOFs | 30 | e-means a0=0.3 B=256 T=3 | 1.0036 | 0.306 | 0.480 | 0.75 | 0.35 | 0.38 | 0.9 |
| PCA 60 EOFs | 30 | e-means a0=0.3 B=256 T=10 | 1.0008 | 0.307 | 0.480 | 0.73 | 0.41 | 0.38 | 0.8 |
| PCA 60 EOFs | 30 | e-means a0=0.3 B=1024 T=3 | 1.0087 | 0.304 | 0.479 | 0.74 | 0.25 | 0.31 | 0.9 |
| PCA 60 EOFs | 30 | e-means a0=0.3 B=1024 T=10 | 1.0034 | 0.306 | 0.481 | 0.75 | 0.30 | 0.36 | 0.6 |
| MSN-mask30 embedding | 10 | Lloyd (sklearn, n_init 10) | 1.0000 | 0.175 | 0.324 | 0.79 | 0.98 | 1.00 | 8.5 |
| MSN-mask30 embedding | 10 | e-means a0=0.03 B=256 T=3 | 1.0367 | 0.169 | 0.319 | 0.78 | 0.47 | 0.55 | 7.1 |
| MSN-mask30 embedding | 10 | e-means a0=0.03 B=256 T=10 | 1.0218 | 0.169 | 0.320 | 0.79 | 0.53 | 0.62 | 5.4 |
| MSN-mask30 embedding | 10 | e-means a0=0.03 B=1024 T=3 | 1.0524 | 0.164 | 0.316 | 0.84 | 0.44 | 0.52 | 6.3 |
| MSN-mask30 embedding | 10 | e-means a0=0.03 B=1024 T=10 | 1.0348 | 0.170 | 0.320 | 0.82 | 0.52 | 0.57 | 6.4 |
| MSN-mask30 embedding | 10 | e-means a0=0.1 B=256 T=3 | 1.0241 | 0.169 | 0.321 | 0.80 | 0.52 | 0.61 | 5.9 |
| MSN-mask30 embedding | 10 | e-means a0=0.1 B=256 T=10 | 1.0128 | 0.175 | 0.322 | 0.79 | 0.57 | 0.69 | 7.0 |
| MSN-mask30 embedding | 10 | e-means a0=0.1 B=1024 T=3 | 1.0423 | 0.170 | 0.319 | 0.83 | 0.47 | 0.54 | 6.3 |
| MSN-mask30 embedding | 10 | e-means a0=0.1 B=1024 T=10 | 1.0215 | 0.174 | 0.321 | 0.81 | 0.61 | 0.63 | 5.7 |
| MSN-mask30 embedding | 10 | e-means a0=0.3 B=256 T=3 | 1.0114 | 0.175 | 0.324 | 0.78 | 0.57 | 0.71 | 6.3 |
| MSN-mask30 embedding | 10 | e-means a0=0.3 B=256 T=10 | 1.0052 | 0.175 | 0.323 | 0.79 | 0.68 | 0.83 | 7.3 |
| MSN-mask30 embedding | 10 | e-means a0=0.3 B=1024 T=3 | 1.0264 | 0.172 | 0.320 | 0.81 | 0.57 | 0.60 | 6.1 |
| MSN-mask30 embedding | 10 | e-means a0=0.3 B=1024 T=10 | 1.0093 | 0.174 | 0.321 | 0.81 | 0.74 | 0.78 | 7.5 |
| MSN-mask30 embedding | 30 | Lloyd (sklearn, n_init 10) | 1.0000 | 0.209 | 0.355 | 0.91 | 1.00 | 1.00 | 2.9 |
| MSN-mask30 embedding | 30 | e-means a0=0.03 B=256 T=3 | 1.1128 | 0.208 | 0.354 | 0.90 | 0.68 | 0.81 | 1.4 |
| MSN-mask30 embedding | 30 | e-means a0=0.03 B=256 T=10 | 1.0693 | 0.208 | 0.355 | 0.90 | 0.78 | 0.87 | 1.3 |
| MSN-mask30 embedding | 30 | e-means a0=0.03 B=1024 T=3 | 1.1788 | 0.209 | 0.355 | 0.91 | 0.57 | 0.71 | 1.4 |
| MSN-mask30 embedding | 30 | e-means a0=0.03 B=1024 T=10 | 1.1395 | 0.209 | 0.356 | 0.92 | 0.64 | 0.77 | 1.4 |
| MSN-mask30 embedding | 30 | e-means a0=0.1 B=256 T=3 | 1.0706 | 0.208 | 0.355 | 0.90 | 0.77 | 0.87 | 1.4 |
| MSN-mask30 embedding | 30 | e-means a0=0.1 B=256 T=10 | 1.0497 | 0.208 | 0.355 | 0.90 | 0.82 | 0.90 | 1.7 |
| MSN-mask30 embedding | 30 | e-means a0=0.1 B=1024 T=3 | 1.1609 | 0.209 | 0.356 | 0.92 | 0.60 | 0.74 | 1.4 |
| MSN-mask30 embedding | 30 | e-means a0=0.1 B=1024 T=10 | 1.0947 | 0.209 | 0.355 | 0.91 | 0.73 | 0.83 | 1.4 |
| MSN-mask30 embedding | 30 | e-means a0=0.3 B=256 T=3 | 1.0504 | 0.208 | 0.356 | 0.90 | 0.82 | 0.90 | 1.3 |
| MSN-mask30 embedding | 30 | e-means a0=0.3 B=256 T=10 | 1.0357 | 0.208 | 0.355 | 0.90 | 0.90 | 0.92 | 1.8 |
| MSN-mask30 embedding | 30 | e-means a0=0.3 B=1024 T=3 | 1.1210 | 0.209 | 0.356 | 0.91 | 0.69 | 0.80 | 1.3 |
| MSN-mask30 embedding | 30 | e-means a0=0.3 B=1024 T=10 | 1.0796 | 0.208 | 0.354 | 0.91 | 0.77 | 0.85 | 1.7 |
