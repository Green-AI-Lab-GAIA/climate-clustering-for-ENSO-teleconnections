## warm days: top 30% of the domain-mean T anomaly (threshold +0.21 sigma, 5378 days); years with most days: 2024 (328), 1998 (272), 1987 (256), 2010 (236), 1983 (230), 2016 (229), 2023 (224), 1980 (185)

### warm / eof: k-means on 40 EOFs (63% var) of [T minus daily domain mean, precip]

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| W0 | 31 | +0.59 | +0.40 (0.002) | +0.40 | +0.34 (0.002) | +0.41 | +0.36 | +0.37 | 1980:85, 2010:82, 1998:78, 1987:77, 2024:70 |
| W1 | 25 | +0.62 | +0.38 (0.002) | +0.40 | +0.41 (0.002) | +0.39 | +0.36 | +0.35 | 2024:160, 2023:118, 2020:64, 1995:57, 1997:57 |
| W2 | 19 | +0.54 | +0.37 (0.002) | +0.37 | +0.30 (0.002) | +0.33 | +0.34 | +0.29 | 1998:70, 1983:64, 1987:64, 2016:52, 2024:52 |
| W3 | 25 | +0.47 | +0.23 (0.002) | +0.23 | +0.29 (0.002) | +0.32 | +0.19 | +0.29 | 1998:100, 1987:80, 2010:73, 2016:71, 2019:67 |

Composition among warm days (310 months with >= 5 warm days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

"+ level": also controlling the mean level of the month's extreme days and of all its days

| sub | mean level | share r ONI given TNA (p) | ... + level (p) | share r TNA given ONI (p) | ... + level (p) |
|---|---|---|---|---|---|
| W0 | +0.59 | -0.02 (0.992) | -0.02 (0.988) | +0.01 (1.000) | +0.00 (1.000) |
| W1 | +0.62 | +0.14 (0.091) | +0.08 (0.561) | +0.17 (0.078) | +0.11 (0.318) |
| W2 | +0.54 | +0.07 (0.674) | +0.02 (0.990) | -0.05 (0.829) | -0.08 (0.557) |
| W3 | +0.47 | -0.19 (0.012) | -0.08 (0.541) | -0.15 (0.113) | -0.06 (0.690) |

### warm / msn: k-means on MSN a100-t30_nomean embeddings (self-supervised, pattern-only)

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| W0 | 23 | +0.61 | +0.44 (0.002) | +0.45 | +0.35 (0.002) | +0.36 | +0.42 | +0.32 | 2024:130, 2023:82, 1983:65, 1988:65, 1998:60 |
| W1 | 29 | +0.60 | +0.41 (0.002) | +0.43 | +0.43 (0.002) | +0.40 | +0.40 | +0.37 | 2016:130, 1998:128, 2024:99, 2010:92, 1987:83 |
| W2 | 26 | +0.53 | +0.19 (0.038) | +0.19 | +0.25 (0.002) | +0.29 | +0.14 | +0.27 | 1983:70, 2024:70, 1987:68, 1997:65, 2023:60 |
| W3 | 21 | +0.47 | +0.17 (0.054) | +0.17 | +0.18 (0.052) | +0.26 | +0.13 | +0.24 | 2002:62, 1980:59, 2010:56, 1987:48, 1995:40 |

Composition among warm days (310 months with >= 5 warm days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

"+ level": also controlling the mean level of the month's extreme days and of all its days

| sub | mean level | share r ONI given TNA (p) | ... + level (p) | share r TNA given ONI (p) | ... + level (p) |
|---|---|---|---|---|---|
| W0 | +0.61 | +0.25 (0.002) | +0.13 (0.169) | +0.13 (0.252) | +0.03 (0.980) |
| W1 | +0.60 | +0.13 (0.161) | +0.01 (0.996) | +0.09 (0.541) | -0.01 (1.000) |
| W2 | +0.53 | -0.13 (0.129) | -0.05 (0.881) | -0.09 (0.559) | -0.02 (0.980) |
| W3 | +0.47 | -0.26 (0.002) | -0.10 (0.400) | -0.13 (0.233) | +0.01 (1.000) |

## cold days: bottom 30% of the domain-mean T anomaly (threshold -0.23 sigma, 4863 days); years with most days: 2000 (249), 1985 (211), 1989 (201), 2008 (185), 1986 (183), 1999 (182), 1984 (180), 2001 (177)

### cold / eof: k-means on 40 EOFs (66% var) of [T minus daily domain mean, precip]

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 13 | -0.79 | -0.07 (0.581) | -0.08 | -0.02 (0.992) | -0.00 | -0.08 | +0.01 | 1990:23, 2013:23, 2000:22, 2003:21, 2004:21 |
| C1 | 19 | -0.47 | -0.29 (0.002) | -0.29 | -0.31 (0.002) | -0.35 | -0.24 | -0.32 | 2014:50, 1992:46, 2012:33, 1989:32, 2000:32 |
| C2 | 27 | -0.47 | -0.34 (0.002) | -0.35 | -0.25 (0.002) | -0.27 | -0.31 | -0.23 | 1985:82, 2000:63, 1999:61, 2008:60, 2011:58 |
| C3 | 41 | -0.46 | -0.37 (0.002) | -0.38 | -0.32 (0.002) | -0.34 | -0.34 | -0.30 | 2000:132, 1989:103, 1984:94, 2001:89, 1985:86 |

Composition among cold days (333 months with >= 5 cold days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

"+ level": also controlling the mean level of the month's extreme days and of all its days

| sub | mean level | share r ONI given TNA (p) | ... + level (p) | share r TNA given ONI (p) | ... + level (p) |
|---|---|---|---|---|---|
| C0 | -0.79 | +0.20 (0.002) | +0.08 (0.390) | +0.28 (0.002) | +0.11 (0.272) |
| C1 | -0.47 | +0.09 (0.342) | +0.04 (0.883) | -0.10 (0.326) | -0.12 (0.175) |
| C2 | -0.47 | -0.10 (0.243) | -0.05 (0.851) | -0.01 (1.000) | +0.09 (0.479) |
| C3 | -0.46 | -0.14 (0.058) | -0.04 (0.883) | -0.17 (0.014) | -0.08 (0.590) |

### cold / msn: k-means on MSN a100-t30_nomean embeddings (self-supervised, pattern-only)

(corr ONI-TNA, detrended monthly: +0.18)

| sub | days % | mean level | r ONI (p FWER) | r ONI detr. | r TNA (p FWER) | r TNA detr. | r ONI given TNA | r TNA given ONI | top years |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 23 | -0.67 | -0.11 (0.195) | -0.11 | -0.10 (0.404) | -0.12 | -0.09 | -0.10 | 2022:41, 2000:40, 2013:36, 2018:36, 1992:35 |
| C1 | 27 | -0.45 | -0.21 (0.002) | -0.23 | -0.30 (0.002) | -0.27 | -0.19 | -0.23 | 2000:124, 1985:100, 2001:86, 1999:79, 1984:75 |
| C2 | 21 | -0.46 | -0.28 (0.002) | -0.29 | -0.20 (0.022) | -0.23 | -0.26 | -0.19 | 1985:48, 2011:44, 1997:41, 2008:39, 1986:38 |
| C3 | 29 | -0.46 | -0.36 (0.002) | -0.36 | -0.22 (0.012) | -0.27 | -0.33 | -0.23 | 1989:79, 2008:70, 2011:66, 1984:61, 2012:58 |

Composition among cold days (333 months with >= 5 cold days): partial r of each sub-regime share with ONI (controlling TNA) and with TNA (controlling ONI), detrended; p = FWER over sub-regimes

"+ level": also controlling the mean level of the month's extreme days and of all its days

| sub | mean level | share r ONI given TNA (p) | ... + level (p) | share r TNA given ONI (p) | ... + level (p) |
|---|---|---|---|---|---|
| C0 | -0.67 | +0.20 (0.002) | +0.07 (0.457) | +0.18 (0.044) | -0.02 (0.992) |
| C1 | -0.45 | +0.04 (0.905) | +0.12 (0.113) | -0.09 (0.535) | -0.02 (0.992) |
| C2 | -0.46 | -0.02 (0.978) | -0.01 (0.990) | +0.03 (0.956) | +0.09 (0.477) |
| C3 | -0.46 | -0.23 (0.002) | -0.18 (0.008) | -0.13 (0.191) | -0.04 (0.924) |

